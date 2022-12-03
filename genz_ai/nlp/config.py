class Config:
    def __init__(
            self,
            input_vocab_size=1000,
            target_vocab_size=100,
            hidden_size=512,
            units=1024,
            dropout_rate=0.2,
            initial_range=0.01,
            hidden_activation='gelu',
            num_hidden_layers=8,
            num_heads=8,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            maxlen=128,
            dff=1024,
            layernorm_epsilon=1e-12,
            num_class=2,
            seq2seq_attention='luong',
            num_lang = 2,
    ):
        self.num_lang = num_lang
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hidden_size = hidden_size
        self.units = units
        self.dropout_rate = dropout_rate
        self.initial_range = initial_range
        self.hidden_activation = hidden_activation
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.maxlen = maxlen
        self.dff = dff
        self.layernorm_epsilon = layernorm_epsilon
        self.num_class = num_class
        self.seq2seq_attention = seq2seq_attention

