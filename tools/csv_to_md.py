import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='csv to md')
    parser.add_argument('--csv_path', type=str, default='./result_thrs0.01.csv')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--indent', type=int, default = 12)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    csv_path = args.csv_path
    save_path = args.save_path
    if save_path == '':
        save_path = csv_path.replace('.csv', '.md')
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    indent = args.indent
    indent_str = ' ' * indent
    header_indent_str = ' ' * (indent - 4)
    
    header = f'{indent_str}| nodule_type | recall(%) | precision(%) | f1 | tp | fp | fn | tn |\n{indent_str}| --- | --- | --- | --- | --- | --- | --- | --- |\n'
    output_lines = []
    for i in range(0, len(lines), 9):
        info = lines[i + 1].split(',')
        iou_threshold, txt_path = info[0], info[1].strip()
        output_lines.append("{}- iou_threshold = {}, txt_path = '{}'\n\n".format(header_indent_str, iou_threshold, os.path.basename(txt_path)))
        output_lines.append(header)
        for j in range(3, 9):
            info = lines[i + j].split(',')
            nodule_type = info[0]
            recall = info[1]
            precision = info[2]
            f1 = info[3]
            tp = info[4]
            fp = info[5]
            fn = info[6]
            tn = info[7].strip()
            line = '{}| {} | {} | {} | {} | {} | {} | {} | {} |\n'.format(indent_str, nodule_type, recall, precision, f1, tp, fp, fn, tn)
            output_lines.append(line)
        output_lines.append('\n')
        
    # Remove the last '\n'
    output_lines = output_lines[:-1]
    output_lines[-1] = output_lines[-1][:-1]
    with open(save_path, 'w') as f:
        f.writelines(output_lines)