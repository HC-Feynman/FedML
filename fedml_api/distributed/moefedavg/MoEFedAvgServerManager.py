import logging
import os
import sys

from .message_define import MyMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    # from FedML.fedml_core.distributed.communication.message import Message
    # from FedML.fedml_core.distributed.server.server_manager import ServerManager
    pass


class MoEFedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False,
                 preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        # global_model_params = self.aggregator.get_global_model_params()
        gating_params = self.aggregator.get_gating_parameters()
        if self.args.is_mobile == 1:
            logging.error("not implemented")
            # global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_gating_config(process_id, gating_params, client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MOE_MODEL_TO_SERVER,
                                              self.handle_message_receive_moe_model_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_EXPERT_ID_TO_SERVER,
                                              self.handle_message_receive_expert_ids_from_client)

    # def handle_message_receive_model_from_client(self, msg_params):
    #     sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
    #     model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
    #     local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
    #
    #     self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
    #     b_all_received = self.aggregator.check_whether_all_receive()
    #     logging.info("b_all_received = " + str(b_all_received))
    #     if b_all_received:
    #         global_model_params = self.aggregator.aggregate()
    #         self.aggregator.test_on_server_for_all_clients(self.round_idx)
    #
    #         # start the next round
    #         self.round_idx += 1
    #         if self.round_idx == self.round_num:
    #             # post_complete_message_to_sweep_process(self.args)
    #             self.finish()
    #             print('here')
    #             return
    #         if self.is_preprocessed:
    #             if self.preprocessed_client_lists is None:
    #                 # sampling has already been done in data preprocessor
    #                 client_indexes = [self.round_idx] * self.args.client_num_per_round
    #             else:
    #                 client_indexes = self.preprocessed_client_lists[self.round_idx]
    #         else:
    #             # sampling clients
    #             client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
    #                                                              self.args.client_num_per_round)
    #
    #         print('indexes of clients: ' + str(client_indexes))
    #         print("size = %d" % self.size)
    #         if self.args.is_mobile == 1:
    #             global_model_params = transform_tensor_to_list(global_model_params)
    #
    #         for receiver_id in range(1, self.size):
    #             self.send_message_sync_model_to_client(receiver_id, global_model_params,
    #                                                    client_indexes[receiver_id - 1])

    def handle_message_receive_moe_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        moe_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MOE_MODEL_PARAMS)
        local_sample_number_of_each_expert = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES_OF_EACH_EXPERT)

        self.aggregator.add_local_trained_result(sender_id - 1, moe_model_params, local_sample_number_of_each_expert)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            gating_params, eid2params = self.aggregator.aggregate()
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                self.finish()
                print('here')
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)

            print('indexes of clients: ' + str(client_indexes))
            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                logging.error("not implemented")
                # global_model_params = transform_tensor_to_list(global_model_params)

            for receiver_id in range(1, self.size):
                self.send_message_sync_gating_model_to_client(receiver_id, gating_params,
                                                              client_indexes[receiver_id - 1])

    def get_experts(self, eid_list):
        eid2params = dict()
        for eid in eid_list:
            params = self.aggregator.model["experts"][eid]
            eid2params[eid] = params.state_dict()
        return eid2params

    def handle_message_receive_expert_ids_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        expert_id_list = msg_params.get(MyMessage.MSG_ARG_KEY_EXPERT_ID_LIST)
        eid2params = self.get_experts(expert_id_list)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_EXPERTS_TO_CLIENT, self.get_sender_id(), sender_id)
        message.add_params(MyMessage.MSG_ARG_KEY_EXPERT_ID_PARAMS_DICT, eid2params)
        self.send_message(message)

    # def send_message_init_config(self, receive_id, global_model_params, client_index):
    #     message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
    #     message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
    #     message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
    #     self.send_message(message)

    def send_message_init_gating_config(self, receive_id, gating_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_GATING_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GATING_MODEL_PARAMS, gating_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    # def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
    #     logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
    #     message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
    #     message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
    #     message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
    #     self.send_message(message)

    def send_message_sync_gating_model_to_client(self, receive_id, gating_model_params, client_index):
        logging.info("send_message_sync_gating_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_GATING_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GATING_MODEL_PARAMS, gating_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
