class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    ### for MoE
    MSG_TYPE_S2C_INIT_GATING_CONFIG = 5
    MSG_TYPE_S2C_SYNC_GATING_MODEL_TO_CLIENT = 6
    MSG_TYPE_S2C_SEND_EXPERTS_TO_CLIENT = 9

    MSG_TYPE_C2S_SEND_EXPERT_ID_TO_SERVER = 7
    MSG_TYPE_C2S_SEND_MOE_MODEL_TO_SERVER = 8  # include the gating and some experts

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    ### for MoE
    MSG_ARG_KEY_GATING_MODEL_PARAMS = "gating_model_params"
    MSG_ARG_KEY_MOE_MODEL_PARAMS = "moe_model_params"
    MSG_ARG_KEY_NUM_SAMPLES_OF_EACH_EXPERT = "num_samples_of_each_expert"  # dict {expert_id: num_samples}
    MSG_ARG_KEY_EXPERT_ID_LIST = "list_of_expert_id"
    MSG_ARG_KEY_EXPERT_ID_PARAMS_DICT = "eid_2_param"


