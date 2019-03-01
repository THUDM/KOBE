def model_opts(parser):
    """
    configuration for training and evaluation
    :param parser: parser
    :return: none
    """
    parser.add_argument('--config', default='default.yaml', type=str,
                        help="config file")
    parser.add_argument('--gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('--restore', default='', type=str,
                        help="restore checkpoint")
    parser.add_argument('--seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('--model', default='seq2seq', type=str,
                        help="Model selection, seq2seq or tensor2tensor")
    parser.add_argument('--mode', default='train', type=str,
                        help="Mode selection")
    parser.add_argument('--module', default='seq2seq', type=str,
                        help="Module selection")
    parser.add_argument('--attention', default='luong', type=str,
                        help="attention selection")
    parser.add_argument('--log', default='', type=str,
                        help="log directory")
    parser.add_argument('--refF', default='', type=str,
                        help="reference")
    parser.add_argument('--save_individual', action='store_true', default=False,
                        help="save individual checkpoint")
    parser.add_argument('--num_processes', type=int, default=4,
                        help="number of processes")
    parser.add_argument('--char', action='store_true', default=False, 
                        help='char level decoding')
    parser.add_argument('--pool_size', type=int, default=0,
                        help="pool size of maxout layer")
    parser.add_argument('--scale', type=float, default=1,
                        help="proportion of the training set")
    parser.add_argument('--max_split', type=int, default=0,
                        help="max generator time steps for memory efficiency")
    parser.add_argument('--split_num', type=int, default=0,
                        help="split number for splitres")
    parser.add_argument('--pretrain', default='', type=str,
                        help="load pretrain encoder")
    parser.add_argument('--label_dict_file', default='', type=str,
                        help="label_dict")
    parser.add_argument('--swish', action='store_true', default=False, 
                        help="swish")
    parser.add_argument('--gate', action='store_true', default=False, 
                        help="to guarantee selfatttn is working for global encoding")
    parser.add_argument('--selfatt', action='store_true', default=False, 
                        help="selfatt for both global encoding and inverse attention")
    parser.add_argument('--Bernoulli', action='store_true', default=False, 
                        help="Bernoulli selection")
    parser.add_argument('--schesamp', action='store_true', default=False, 
                        help="schedule sampling")
    parser.add_argument('--schedule', action='store_true', default=False, 
                        help="learning rate schedule")
    parser.add_argument('--rl', action='store_true', default=False, 
                        help="reinforcement learning")
    parser.add_argument('--epoch_decay', action='store_true', default=False, 
                        help="decay by epochs after decay starts")

    # for prediction
    parser.add_argument('--test_src', default='', type=str,
                        help="test source file")
    parser.add_argument('--test_tgt', default='', type=str,
                        help="test target file")
    parser.add_argument('--src_filter', type=int, default=0,
                        help="Maximum source sequence length")
    parser.add_argument('--tgt_filter', type=int, default=0,
                        help="Maximum target sequence length")
    parser.add_argument('--src_trun', type=int, default=0,
                        help="Truncate source sequence length")
    parser.add_argument('--tgt_trun', type=int, default=0,
                        help="Truncate target sequence length")
    parser.add_argument('--lower', action='store_true',
                        help='lower the case')


def convert_to_config(opt, config):
    """
    function to combine opt and config
    :param opt: commands
    :param config: yaml configuration
    :return: none
    """
    opt = vars(opt)
    for key in opt:
        if key not in config:
            config[key] = opt[key]