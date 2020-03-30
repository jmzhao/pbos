import logging

def add_logging_args(parser):
    group = parser.add_argument_group('logging arguments')
    group.add_argument('--log_level', default='INFO',
        help='log level used by logging module')
    return parser

def logging_config(args):
    """Set log level using args.log_level"""
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)
