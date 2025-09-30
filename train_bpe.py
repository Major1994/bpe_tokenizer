import re
from collections import Counter
from typing import Dict, Tuple, List, Set
import argparse


def read_sentences(filename: str) -> Counter:
    """
    读取文本文件，构建初始词汇表。
    每个词被拆分为字符，并在末尾添加 </w> 表示词尾。
    """
    vocab = Counter()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                words = line.split()
                for word in words:
                    if word:  # 避免空字符串
                        # 将词转为 "c h a r s </w>"
                        tokenized = ' '.join(list(word)) + ' </w>'
                        vocab[tokenized] += 1
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {filename}")
    except Exception as e:
        raise RuntimeError(f"读取文件时出错: {e}")

    if not vocab:
        raise ValueError("文件为空或未解析到任何词汇")

    return vocab


def get_stats(vocab: Counter) -> Dict[Tuple[str, str], int]:
    """
    统计词汇表中所有相邻符号对的频率。
    """
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq
    return pairs


def merge_vocab(pair: Tuple[str, str], vocab: Counter) -> Counter:
    """
    将词汇表中指定的字符对进行合并。
    """
    vocab_new = Counter()
    # 转义字符对，避免正则特殊字符问题
    bigram = ' '.join(pair)
    escaped_bigram = re.escape(bigram)
    # 使用负向断言确保只匹配独立的符号对
    pattern = re.compile(r'(?<!\S)' + escaped_bigram + r'(?!\S)')

    for word, freq in vocab.items():
        # 合并字符对（如 'l o' -> 'lo'）
        new_word = pattern.sub(''.join(pair), word)
        if new_word != word:  # 仅当发生替换时才更新
            vocab_new[new_word] = freq
        else:
            vocab_new[word] = freq  # 未变化则保留原词
    return vocab_new


def learn_bpe(corpus_file: str, num_merges: int, output_vocab: str):
    """
    主函数：学习 BPE 合并规则。
    """
    print(f"正在读取语料: {corpus_file}")
    vocab = read_sentences(corpus_file)
    print(f"初始词汇量: {len(vocab)}")

    bpe_merges = []  # 存储学习到的合并规则

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            print(f"在第 {i + 1} 轮后，已无更多可合并的对。")
            break

        # 找到最高频的字符对
        best_pair = max(pairs, key=pairs.get)
        bpe_merges.append(best_pair)

        # 执行合并
        vocab = merge_vocab(best_pair, vocab)

        if (i + 1) % 100 == 0 or i == num_merges - 1:
            print(f"第 {i + 1}/{num_merges} 次合并: {best_pair} (频率: {pairs[best_pair]})")

    # 保存合并规则
    with open(output_vocab, 'w', encoding='utf-8') as f:
        for pair in bpe_merges:
            f.write(f"{pair[0]} {pair[1]}\n")
    print(f"BPE 合并规则已保存至: {output_vocab}")

    return bpe_merges, vocab


def main():
    parser = argparse.ArgumentParser(description="BPE 子词分词训练")
    parser.add_argument("--input", type=str, default="news.txt", help="输入语料文件路径")
    parser.add_argument("--merges", type=int, default=2000, help="合并次数")
    parser.add_argument("--output", type=str, default="bpe_codes.txt", help="输出 BPE 规则文件")

    args = parser.parse_args()

    learn_bpe(args.input, args.merges, args.output)


if __name__ == '__main__':
    main()
