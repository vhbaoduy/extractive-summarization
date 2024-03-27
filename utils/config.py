class ModelConfig:
    def __init__(self,
                 extractor="HER",
                 word_input_size=100,
                 sentence_input_size=400,
                 word_lstm_hidden_units=200,
                 sentence_lstm_hidden_units=200,
                 num_filters=100,
                 kernel_sizes=[1, 2, 3],
                 dropout=0.0,
                 num_lstm_layers=2,
                 decode_hidden_units=200,
                 **kwargs):
        self.extractor = extractor
        self.word_input_size = word_input_size
        self.sentence_input_size = sentence_input_size
        self.word_lstm_hidden_units = word_lstm_hidden_units
        self.sentence_lstm_hidden_units = sentence_lstm_hidden_units
        self.num_filters = num_filters  # feature maps
        self.kernel_sizes = kernel_sizes  # H corresponding to K= 3
        self.dropout = dropout
        self.num_lstm_layers = num_lstm_layers
        self.decode_hidden_units = decode_hidden_units


class ReinforceConfig:
    def __init__(self,
                 B=20,
                 num_of_min_sents=1,
                 num_of_max_sents=3,
                 rl_baseline_method="batch_avg",
                 rouge_metric="avg_f",
                 oracle_length=-1,
                 length_limit=-1,
                 std_rouge=False,
                 **kwargs):
        self.B = B
        self.num_of_min_sents = num_of_min_sents
        self.num_of_max_sents = num_of_max_sents
        self.rl_baseline_method = rl_baseline_method
        self.rouge_metric = rouge_metric
        self.oracle_length = oracle_length
        self.length_limit = length_limit
        self.std_rouge= std_rouge

class OptConfig:
    def __init__(self,
                 epochs=2,
                 lr=1e-5,
                 beta=(0, 0.999),
                 weight_decay=1e-5,
                 eval_steps=5000,
                 print_steps=200,
                 **kwargs) -> None:
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

        self.eval_steps = eval_steps
        self.print_steps = print_steps


class Configs(dict):
    def __init__(self,
                 model: ModelConfig,
                 reinforce: ReinforceConfig,
                 optimize: OptConfig,
                 log_path="logs",
                 save_path="models",
                 result_path="results",
                 pretrained_model_path=None,
                 **kwargs):
        self.model = ModelConfig(**model) if isinstance(model, dict) else model
        self.reinforce = ReinforceConfig(
            **reinforce) if isinstance(reinforce, dict) else reinforce
        self.optimize = OptConfig(
            **optimize) if isinstance(optimize, dict) else optimize
        self.log_path = log_path
        self.save_path = save_path
        self.result_path = result_path
        self.pretrained_model_path = pretrained_model_path
    # # Opimitize
    # lr = 1e-5
    # beta = (0, 0.999)
    # weight_decay = 1e-6
    # epochs = 1

    # rl_baseline_method: str = "batch_avg"
    # oracle_length: int = -1
    # length_limit: int = -1

    # eval_steps = 5000
    # print_steps = 100
