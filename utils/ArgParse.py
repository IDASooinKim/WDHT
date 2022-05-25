import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description='Argment for WDHT model')
    
    parser.add_argument('--epochs', '-e', dest='epochs', 
            metavar='E', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--learning_rate', '-l', dest='learning_rate', metavar='LR', 
            type=float, default=0.0001, help='Number of Learning Rate')
    parser.add_argument('--decay', '-dc', dest='decay', 
            metavar='D', type=float, default=0.9)
    parser.add_argument('--hash_size', dest='hash_size', 
            metavar='H', type=int, default=32,help='hash size')
    parser.add_argument('--batch_size', '-b', dest='batch_size', 
            metavar='B', type=int, default=64, help='Number of Batch Size')
    parser.add_argument('--img_path', '-ip', dest='img_path', 
            metavar='IP', type=str, default='/data/tag_test')
    parser.add_argument('--label_path', '-lp', dest='label_path', 
            metavar='LP', 
            type=str, default='/data/tag_test/tag_sample_annotation')
    parser.add_argument('--class_num', '-cn', dest='class_num', 
            metavar='CN', type=int, default=374, help='Size of Classify Layer')
    parser.add_argument('--device', metavar='-d', type=str, default='cpu', 
            help='Device for load model')
    parser.add_argument('--save_path', '-mp', dest='save_path', metavar='SP', type=str, 
            default='result/', help='Path for Saving Model')
    parser.add_argument('--w2v_path', '-w', dest='w2v_path', metavar='W', type=str, 
            default='word2vec_saved.model', help='Path for Word2Vector model')
    parser.add_argument('--test_ratio', '-t', dest='test_ratio', metavar='T', 
            type=str, default=0.2, help='Ratio of test datasets')
    
    return parser.parse_args()
