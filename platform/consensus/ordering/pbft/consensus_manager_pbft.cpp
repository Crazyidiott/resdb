/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "platform/consensus/ordering/pbft/consensus_manager_pbft.h"

#include <glog/logging.h>
#include <unistd.h>

#include "common/crypto/signature_verifier.h"

namespace resdb {

ConsensusManagerPBFT::ConsensusManagerPBFT(
    const ResDBConfig& config, std::unique_ptr<TransactionManager> executor,
    std::unique_ptr<CustomQuery> query_executor)
    : ConsensusManager(config),
      system_info_(std::make_unique<SystemInfo>(config)),
      checkpoint_manager_(std::make_unique<CheckPointManager>(
          config, GetBroadCastClient(), GetSignatureVerifier())),
      message_manager_(std::make_unique<MessageManager>(
          config, std::move(executor), checkpoint_manager_.get(),
          system_info_.get())),
      commitment_(std::make_unique<Commitment>(config_, message_manager_.get(),
                                               GetBroadCastClient(),
                                               GetSignatureVerifier())),
      query_(std::make_unique<Query>(config_, message_manager_.get(),
                                     std::move(query_executor))),
      response_manager_(config_.IsPerformanceRunning()
                            ? nullptr
                            : std::make_unique<ResponseManager>(
                                  config_, GetBroadCastClient(),
                                  system_info_.get(), GetSignatureVerifier())),
      performance_manager_(config_.IsPerformanceRunning()
                               ? std::make_unique<PerformanceManager>(
                                     config_, GetBroadCastClient(),
                                     system_info_.get(), GetSignatureVerifier())
                               : nullptr),
      view_change_manager_(std::make_unique<ViewChangeManager>(
          config_, checkpoint_manager_.get(), message_manager_.get(),
          system_info_.get(), GetBroadCastClient(), GetSignatureVerifier())),
      recovery_(std::make_unique<Recovery>(config_, checkpoint_manager_.get(),
                                           system_info_.get(),
                                           message_manager_->GetStorage())) {
  LOG(INFO) << "is running is performance mode:"
            << config_.IsPerformanceRunning();
  global_stats_ = Stats::GetGlobalStats();

  view_change_manager_->SetDuplicateManager(commitment_->GetDuplicateManager());

  // 添加：从配置中读取pipeline大小（如果配置支持的话）
  // 如果配置不支持，使用默认值100
  // if (config_.GetConfigData().has_max_pipeline_size()) {
  //   max_pipeline_size_ = config_.GetConfigData().max_pipeline_size();
  // }
  
  // 设置执行完成的回调
  message_manager_->SetExecutionCallback(
      [this](uint64_t seq) { this->OnRequestExecuted(seq); });

  recovery_->ReadLogs(
      [&](const SystemInfoData& data) {
        system_info_->SetCurrentView(data.view());
        system_info_->SetPrimary(data.primary_id());
      },
      [&](std::unique_ptr<Context> context, std::unique_ptr<Request> request) {
        return InternalConsensusCommit(std::move(context), std::move(request));
      });
}

void ConsensusManagerPBFT::SetNeedCommitQC(bool need_qc) {
  commitment_->SetNeedCommitQC(need_qc);
}

void ConsensusManagerPBFT::Start() { ConsensusManager::Start(); }

std::vector<ReplicaInfo> ConsensusManagerPBFT::GetReplicas() {
  return message_manager_->GetReplicas();
}

uint32_t ConsensusManagerPBFT::GetPrimary() {
  return system_info_->GetPrimaryId();
}

uint32_t ConsensusManagerPBFT::GetVersion() {
  return system_info_->GetCurrentView();
}

void ConsensusManagerPBFT::SetPrimary(uint32_t primary, uint64_t version) {
  if (version > system_info_->GetCurrentView()) {
    system_info_->SetCurrentView(version);
    system_info_->SetPrimary(primary);
  }
}

void ConsensusManagerPBFT::AddPendingRequest(std::unique_ptr<Context> context,
                                             std::unique_ptr<Request> request) {
  std::lock_guard<std::mutex> lk(mutex_);
  request_pending_.push(std::make_pair(std::move(context), std::move(request)));
}

void ConsensusManagerPBFT::AddComplainedRequest(
    std::unique_ptr<Context> context, std::unique_ptr<Request> request) {
  std::lock_guard<std::mutex> lk(mutex_);
  request_complained_.push(
      std::make_pair(std::move(context), std::move(request)));
}

absl::StatusOr<std::pair<std::unique_ptr<Context>, std::unique_ptr<Request>>>
ConsensusManagerPBFT::PopPendingRequest() {
  std::lock_guard<std::mutex> lk(mutex_);
  if (request_pending_.empty()) {
    // LOG(ERROR) << "empty:";
    return absl::InternalError("No Data.");
  }
  auto new_request = std::move(request_pending_.front());
  request_pending_.pop();
  return new_request;
}

absl::StatusOr<std::pair<std::unique_ptr<Context>, std::unique_ptr<Request>>>
ConsensusManagerPBFT::PopComplainedRequest() {
  std::lock_guard<std::mutex> lk(mutex_);
  if (request_complained_.empty()) {
    // LOG(ERROR) << "empty:";
    return absl::InternalError("No Data.");
  }
  auto new_request = std::move(request_complained_.front());
  request_complained_.pop();
  return new_request;
}

// The implementation of PBFT.
int ConsensusManagerPBFT::ConsensusCommit(std::unique_ptr<Context> context,
                                          std::unique_ptr<Request> request) {
  // LOG(INFO) << "recv impl type:" << request->type() << " "
  //          << "sender id:" << request->sender_id();
  // If it is in viewchange, push the request to the queue
  // for the requests from the new view which come before
  // the local new view done.
  recovery_->AddRequest(context.get(), request.get());
  if (config_.GetConfigData().enable_viewchange()) {
    view_change_manager_->MayStart();
    if (view_change_manager_->IsInViewChange()) {
      // 添加：视图变更时，将waiting_queue移到pending_queue
      MoveWaitingToPending();

      switch (request->type()) {
        case Request::TYPE_NEW_TXNS:
        case Request::TYPE_PRE_PREPARE:
        case Request::TYPE_PREPARE:
        case Request::TYPE_COMMIT:
          AddPendingRequest(std::move(context), std::move(request));
          return 0;
      }
    } else {
      while (true) {
        auto new_request = PopPendingRequest();
        if (!new_request.ok()) {
          break;
        }
        InternalConsensusCommit(std::move((*new_request).first),
                                std::move((*new_request).second));
      }
    }
  }
  
  /* 添加代码，控制pipeline
      逻辑1：收到request之后，先不处理，放入额外的队列中，记录为waiting queue
      逻辑2：根据设置的pipeline参数n，一次允许处理n个request
      逻辑3：当有request处理完成之后（完成commit并执行），再从队列中取出一个request进行处理，始终保持当前正在处理的request数量不超过n
      逻辑4：触发view change的时候，需要把waiting queue中所有的request加入到pending queue中，通过AddPendingRequest方法
  */


  int ret = InternalConsensusCommit(std::move(context), std::move(request));
  if (config_.GetConfigData().enable_viewchange()) {
    if (ret == -4) {
      while (true) {
        auto new_request = PopComplainedRequest();
        if (!new_request.ok()) {
          break;
        }
        // LOG(ERROR) << "[POP COMPLAINED REQUEST]";
        InternalConsensusCommit(std::move((*new_request).first),
                                std::move((*new_request).second));
      }
    }
  }
  return ret;
}

int ConsensusManagerPBFT::InternalConsensusCommit(
    std::unique_ptr<Context> context, std::unique_ptr<Request> request) {
  // LOG(INFO) << "recv impl type:" << request->type() << " "
  //         << "sender id:" << request->sender_id()<<" seq:"<<request->seq();

  switch (request->type()) {
    case Request::TYPE_CLIENT_REQUEST:
      if (config_.IsPerformanceRunning()) {
        return performance_manager_->StartEval();
      }
      return response_manager_->NewUserRequest(std::move(context),
                                               std::move(request));
    case Request::TYPE_RESPONSE:
      if (config_.IsPerformanceRunning()) {
        return performance_manager_->ProcessResponseMsg(std::move(context),
                                                        std::move(request));
      }
      return response_manager_->ProcessResponseMsg(std::move(context),
                                                   std::move(request));
    // case Request::TYPE_NEW_TXNS: {
    //   LOG(ERROR) << "TESTATATSTAETSETA";

    //   uint64_t proxy_id = request->proxy_id();
    //   std::string hash = request->hash();
    //   int ret = commitment_->ProcessNewRequest(std::move(context),
    //                                            std::move(request));
    //   if (ret == -3) {
    //     LOG(ERROR) << "BAD RETURN";
    //     std::pair<std::unique_ptr<Context>, std::unique_ptr<Request>>
    //         request_complained;
    //     {
    //       std::lock_guard<std::mutex> lk(commitment_->rc_mutex_);

    //       request_complained =
    //           std::move(commitment_->request_complained_.front());
    //       commitment_->request_complained_.pop();
    //     }
    //     AddComplainedRequest(std::move(request_complained.first),
    //                          std::move(request_complained.second));
    //     view_change_manager_->AddComplaintTimer(proxy_id, hash);
    //   }
    //   return ret;
    // }
    case Request::TYPE_NEW_TXNS: {
      uint64_t proxy_id = request->proxy_id();
      std::string hash = request->hash();
      
      // Pipeline控制：只在Primary节点生效
      if (IsPrimary() && !CanProcessNewRequest()) {
        AddToWaitingQueue(std::move(context), std::move(request));
        return 0;  // 请求已加入等待队列，暂不处理
      }
      
      int ret = commitment_->ProcessNewRequest(std::move(context),
                                              std::move(request));
      
      // 如果成功开始处理，增加计数
      if (ret == 0 && IsPrimary()) {
        processing_count_.fetch_add(1, std::memory_order_relaxed);
      }
      
      if (ret == -3) {
        LOG(ERROR) << "BAD RETURN";
        std::pair<std::unique_ptr<Context>, std::unique_ptr<Request>>
            request_complained;
        {
          std::lock_guard<std::mutex> lk(commitment_->rc_mutex_);
          request_complained =
              std::move(commitment_->request_complained_.front());
          commitment_->request_complained_.pop();
        }
        AddComplainedRequest(std::move(request_complained.first),
                            std::move(request_complained.second));
        view_change_manager_->AddComplaintTimer(proxy_id, hash);
      }
      return ret;
    }
    case Request::TYPE_PRE_PREPARE:
      return commitment_->ProcessProposeMsg(std::move(context),
                                            std::move(request));
    case Request::TYPE_PREPARE:
      return commitment_->ProcessPrepareMsg(std::move(context),
                                            std::move(request));
    case Request::TYPE_COMMIT:
      return commitment_->ProcessCommitMsg(std::move(context),
                                           std::move(request));
    case Request::TYPE_CHECKPOINT:
      return checkpoint_manager_->ProcessCheckPoint(std::move(context),
                                                    std::move(request));
    case Request::TYPE_VIEWCHANGE:
      return view_change_manager_->ProcessViewChange(std::move(context),
                                                     std::move(request));
    case Request::TYPE_NEWVIEW:
      return view_change_manager_->ProcessNewView(std::move(context),
                                                  std::move(request));
    case Request::TYPE_QUERY:
      return query_->ProcessQuery(std::move(context), std::move(request));
    case Request::TYPE_REPLICA_STATE:
      return query_->ProcessGetReplicaState(std::move(context),
                                            std::move(request));
    case Request::TYPE_CUSTOM_QUERY:
      return query_->ProcessCustomQuery(std::move(context), std::move(request));
  }
  return 0;
}

void ConsensusManagerPBFT::SetupPerformanceDataFunc(
    std::function<std::string()> func) {
  performance_manager_->SetDataFunc(func);
}

void ConsensusManagerPBFT::SetPreVerifyFunc(
    std::function<bool(const Request&)> func) {
  commitment_->SetPreVerifyFunc(func);
}

// 待审查
// 判断是否为Primary节点
bool ConsensusManagerPBFT::IsPrimary() const {
  return config_.GetSelfInfo().id() == system_info_->GetPrimaryId();
}

// 检查是否可以处理新请求
bool ConsensusManagerPBFT::CanProcessNewRequest() {
  return processing_count_.load(std::memory_order_relaxed) < max_pipeline_size_;
}

// 添加请求到等待队列
void ConsensusManagerPBFT::AddToWaitingQueue(
    std::unique_ptr<Context> context, 
    std::unique_ptr<Request> request) {
  std::lock_guard<std::mutex> lk(pipeline_mutex_);
  waiting_queue_.push(std::make_pair(std::move(context), std::move(request)));
  
  LOG(INFO) << "Request added to waiting queue. Queue size: " 
            << waiting_queue_.size()
            << ", Processing: " << processing_count_.load();
}

// 尝试从等待队列处理请求
void ConsensusManagerPBFT::TryProcessFromWaitingQueue() {
  std::unique_ptr<Context> context;
  std::unique_ptr<Request> request;
  
  {
    std::lock_guard<std::mutex> lk(pipeline_mutex_);
    if (waiting_queue_.empty() || !CanProcessNewRequest()) {
      return;
    }
    
    auto item = std::move(waiting_queue_.front());
    waiting_queue_.pop();
    context = std::move(item.first);
    request = std::move(item.second);
  }
  
  // 递增计数（在锁外进行）
  processing_count_.fetch_add(1, std::memory_order_relaxed);
  
  // 处理请求（注意：这里直接调用commitment，跳过了TYPE_NEW_TXNS的pipeline检查）
  int ret = commitment_->ProcessNewRequest(std::move(context), std::move(request));
  
  if (ret != 0) {
    // 如果处理失败，需要减少计数
    processing_count_.fetch_sub(1, std::memory_order_relaxed);
    // 可能需要继续尝试处理下一个
    TryProcessFromWaitingQueue();
  }
}

// 请求执行完成的回调
void ConsensusManagerPBFT::OnRequestExecuted(uint64_t seq) {
  if (!IsPrimary()) {
    return;
  }
  
  int prev_count = processing_count_.fetch_sub(1, std::memory_order_relaxed);
  
  LOG(INFO) << "Request executed. Seq: " << seq 
            << ", Processing count: " << (prev_count - 1);
  
  // 尝试处理等待队列中的请求
  TryProcessFromWaitingQueue();
}

// 将等待队列中的所有请求移到pending队列
void ConsensusManagerPBFT::MoveWaitingToPending() {
  std::lock_guard<std::mutex> lk(pipeline_mutex_);
  
  while (!waiting_queue_.empty()) {
    auto item = std::move(waiting_queue_.front());
    waiting_queue_.pop();
    AddPendingRequest(std::move(item.first), std::move(item.second));
  }
  
  // 重置处理计数
  processing_count_.store(0, std::memory_order_relaxed);
  
  LOG(INFO) << "Moved all waiting requests to pending queue due to view change";
}

}  // namespace resdb
