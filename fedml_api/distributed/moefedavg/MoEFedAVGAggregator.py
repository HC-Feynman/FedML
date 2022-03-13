import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor


class MoEFedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()  # index: moe_params, where params include gating and experts
        self.sample_num_dict = dict()   # index: sample_num_of_each_expert, sample_num_of_each_expert is also a dict
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    # def get_global_model_params(self):
    #     return self.trainer.get_model_params()

    def get_gating_parameters(self):
        return self.trainer.get_gating_params()

    # def set_global_model_params(self, model_parameters):
    #     self.trainer.set_model_params(model_parameters)

    def set_gating_parameters(self, gating_params):
        return self.trainer.set_gating_params(gating_params)

    def set_experts_parameters(self, eid2params):
        return self.trainer.set_experts_parameters(eid2params)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()

        gating_model_params_list = []
        eid_2_params_list = dict()  # expert id : a list of (workerid, #sample this w this e, params)
        worker_2_num_sample = dict()
        training_num = 0
        for idx in range(self.worker_num):

            local_num_sample = sum(self.sample_num_dict[idx].values()) // self.args.num_experts_to_choose
            worker_2_num_sample[idx] = local_num_sample
            training_num += local_num_sample

            gating_model_params_list.append(self.model_dict[idx]["gating"])
            for eid, eparams in self.model_dict[idx]["experts"].items():
                if eid not in eid_2_params_list:
                    eid_2_params_list[eid] = []
                eid_2_params_list[eid].append((idx, self.sample_num_dict[idx][eid], eparams))

        # aggregate gating
        normal = gating_model_params_list[0]["noise"]
        averaged_feature_map = gating_model_params_list[0]["feature_map"]
        for k in averaged_feature_map.keys():
            for i in range(len(gating_model_params_list)):
                local_sample_number = worker_2_num_sample[i]
                local_gating_feature_map = gating_model_params_list[i]["feature_map"]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_feature_map[k] = local_gating_feature_map[k] * w
                else:
                    averaged_feature_map[k] += local_gating_feature_map[k] * w
        averaged_w_gate = gating_model_params_list[0]["w_gate"]
        averaged_w_noise = gating_model_params_list[0]["w_noise"]
        for i in range(len(gating_model_params_list)):
            local_sample_number = worker_2_num_sample[i]
            w = local_sample_number / training_num
            lcoal_w_gate = gating_model_params_list[i]["w_gate"]
            local_w_noise = gating_model_params_list[i]["w_noise"]
            if i == 0:
                averaged_w_gate = lcoal_w_gate * w
                averaged_w_noise = local_w_noise * w
            else:
                averaged_w_gate += lcoal_w_gate * w
                averaged_w_noise += local_w_noise * w
        gating_params = {"feature_map": averaged_feature_map, "w_gate": averaged_w_gate, "w_noise": averaged_w_noise, "normal": normal}
        self.set_gating_parameters(gating_params)

        # aggregate experts
        eid_2_aggregated_params = dict()
        for eid, eparams_list in eid_2_params_list.items():
            assert len(eparams_list) > 0
            num_samples_this_expert = 0
            for (wid, n_sam_w_e, eparams) in eparams_list:
                num_samples_this_expert += n_sam_w_e
            avg_eparams = eparams_list[0][1]
            for i, (wid, n_sam_w_e, eparams) in enumerate(eparams_list):
                weight = n_sam_w_e / num_samples_this_expert
                for k in avg_eparams.keys():
                    if i == 0:
                        avg_eparams[k] = eparams[k] * weight
                    else:
                        avg_eparams[k] += eparams[k] * weight
            eid_2_aggregated_params[eid] = avg_eparams
        self.set_experts_parameters(eid_2_aggregated_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return gating_params, eid_2_aggregated_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            # the above returns False directly
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
