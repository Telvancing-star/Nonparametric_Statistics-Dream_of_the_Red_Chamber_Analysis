"""
《红楼梦》作者争议分析
----------------------
研究问题：前 80 回与后 40 回用词风格是否存在差异？

流程概览：
  1. 文本预处理与特征构建
  2. 描述性统计与预处理可视化
  3. 全特征非参数检验（Mann-Whitney + Bonferroni/FDR + 代表词 KS 检验）
  4. PCA 探索性可视化 + PC1 补充检验（Mann-Whitney / KS / 置换检验）
  5. 输出统计报告与图表
"""

import itertools
import json
import os
import re

import jieba
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats
from sklearn import decomposition
from tqdm import tqdm

# =============================================================================
# 全局配置
# =============================================================================
V = 2000                  # 每章特征维度：取高频非人名词 Top V
N_FRONT = 80              # 前 80 回为对照组，后 40 回为比较组
N_TOP_WORDS_PLOT = 20     # 描述性统计：高频词条形图展示数量
N_REP_FEATURES = 6        # 代表词数量（用于箱线图/密度图）
N_COMPONENTS = 3          # PCA 保留主成分数
N_PERM = 10000            # PC1 置换检验重复次数
SCREE_COMPONENTS = 10     # 碎石图展示的主成分个数
ALPHA = 0.05              # 显著性水平

OUTPUT_DIR = 'output'
REPORT_PATH = os.path.join(OUTPUT_DIR, 'report.txt')
CHAPTER_DIR = '红楼梦'
USERDICT_PATH = 'userdict.json'
STOPWORDS_PATH = 'stopwords.txt'
SHOW_PLOTS = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 第一部分：文本预处理与特征构建
# =============================================================================
def load_names(file_path=USERDICT_PATH):
    """加载人物别名表，并注册到 jieba 词典（标记为人名 nr）。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        names = json.load(f)
    for name in names:
        for alias_name in names[name]:
            jieba.add_word(alias_name, tag='nr')
    return names


def load_stopwords(file_path=STOPWORDS_PATH):
    """加载停用词表。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip('\n') for line in f.readlines()]


def tokenize_chapter(file_path, stopwords):
    """对单章文本分词，过滤单字词与停用词，返回段落级词列表。"""
    with open(file_path, 'rb') as f:
        content = f.read()
    para = content.decode('utf-8', 'ignore').split('\n')[:-1]
    result = []
    for p in para:
        words = []
        for x in jieba.cut(p, cut_all=False):
            if len(x) <= 1 or x in stopwords:
                continue
            words.append(x)
        result.append(words)
    return result


def tokenize_all_chapters(chapter_dir=CHAPTER_DIR):
    """遍历各章分词，返回 {章节号: [[段1词列表], ...]}。"""
    stopwords = load_stopwords()
    result = {}
    for file_name in tqdm(os.listdir(chapter_dir), desc='分词'):
        index = int(file_name.split('_')[0])
        result[index] = tokenize_chapter(os.path.join(chapter_dir, file_name), stopwords)
    return result


def build_word_freq(tokenized):
    """统计全书词频与各章词频。"""
    artical_word_dict = {}
    chapters_word_dict = {}
    for index in tokenized:
        chapter_word_list = list(itertools.chain.from_iterable(tokenized[index]))
        chapter_word_dict = {}
        for word in chapter_word_list:
            chapter_word_dict[word] = chapter_word_dict.get(word, 0) + 1
            artical_word_dict[word] = artical_word_dict.get(word, 0) + 1
        chapters_word_dict[index] = chapter_word_dict
    sorted_word_list = [
        word for _, word in sorted(
            zip(artical_word_dict.values(), artical_word_dict.keys()), reverse=True)
    ]
    return chapters_word_dict, sorted_word_list, artical_word_dict


def remove_person_names(sorted_word_list, names):
    """从高频词列表中剔除人物名，避免人名主导风格特征。"""
    all_names = [alias for aliases in names.values() for alias in aliases]
    filtered = []
    for word in sorted_word_list:
        is_name = any(re.search(re.compile(f'(.*){name}(.*)'), word) for name in all_names)
        if not is_name:
            filtered.append(word)
    return filtered


def build_feature_matrix(chapters_word_dict, word_list):
    """
    构建章节 × 词语 特征矩阵（120 × V）。
    行按章节号 1–120 排序；列按 word_list 顺序；值为归一化词频。
    """
    chapter_ids = sorted(chapters_word_dict.keys())
    features = np.zeros((len(chapter_ids), len(word_list)))
    for i, chap_id in enumerate(chapter_ids):
        chapter_word_dict = chapters_word_dict[chap_id]
        for j, word in enumerate(word_list):
            features[i, j] = chapter_word_dict.get(word, 0)
    # 按列归一化到 [0, 1]，消除词频绝对量级差异
    col_max = features.max(axis=0)
    col_max[col_max == 0] = 1
    features /= col_max
    return features, chapter_ids


def compute_chapter_stats(tokenized):
    """汇总各章分词后的基本计数指标。"""
    chapter_stats = []
    for chap_id in sorted(tokenized.keys()):
        words = list(itertools.chain.from_iterable(tokenized[chap_id]))
        chapter_stats.append({
            'chapter_id': chap_id,
            'token_count': len(words),
            'unique_count': len(set(words)),
            'paragraph_count': len(tokenized[chap_id]),
        })
    return chapter_stats


def prepare_features():
    """预处理流水线：分词 → 词频 → 去人名 → 特征矩阵。"""
    names = load_names()
    tokenized = tokenize_all_chapters()
    chapters_word_dict, sorted_word_list, artical_word_dict = build_word_freq(tokenized)
    filtered_words = remove_person_names(sorted_word_list, names)
    word_list = filtered_words[:V]
    features, chapter_ids = build_feature_matrix(chapters_word_dict, word_list)
    chapter_stats = compute_chapter_stats(tokenized)
    preprocess_meta = {
        'chapter_stats': chapter_stats,
        'book_word_freq': artical_word_dict,
        'sorted_word_list': sorted_word_list,
        'filtered_words': filtered_words,
        'vocab_before_filter': len(sorted_word_list),
        'vocab_after_filter': len(filtered_words),
        'names_count': sum(len(aliases) for aliases in names.values()),
    }
    return features, word_list, chapter_ids, preprocess_meta


# =============================================================================
# 第二部分：描述性统计与预处理可视化
# =============================================================================
def _save_fig(name):
    """保存当前图像到 output/ 目录。"""
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _group_split(values, n_front=N_FRONT):
    """将按章节排序的一维数组拆分为前 80 回与后 40 回。"""
    return values[:n_front], values[n_front:]


def _describe_array(arr):
    """计算一组样本的常见描述统计量。"""
    arr = np.asarray(arr, dtype=float)
    return {
        'n': len(arr),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
    }


def compute_descriptive_stats(preprocess_meta, features, word_list, chapter_ids, n_front=N_FRONT):
    """汇总预处理后的描述性统计结果。"""
    chapter_stats = preprocess_meta['chapter_stats']
    token_counts = np.array([s['token_count'] for s in chapter_stats], dtype=float)
    unique_counts = np.array([s['unique_count'] for s in chapter_stats], dtype=float)
    para_counts = np.array([s['paragraph_count'] for s in chapter_stats], dtype=float)

    front_tokens, back_tokens = _group_split(token_counts, n_front)
    front_unique, back_unique = _group_split(unique_counts, n_front)
    front_para, back_para = _group_split(para_counts, n_front)

    nonzero_ratio = float(np.mean(features > 0))

    top_words = []
    for word in preprocess_meta['filtered_words'][:N_TOP_WORDS_PLOT]:
        top_words.append({
            'word': word,
            'freq': preprocess_meta['book_word_freq'].get(word, 0),
        })

    return {
        'n_chapters': len(chapter_ids),
        'n_features': len(word_list),
        'total_tokens': int(token_counts.sum()),
        'book_vocab': len(preprocess_meta['book_word_freq']),
        'vocab_before_filter': preprocess_meta['vocab_before_filter'],
        'vocab_after_filter': preprocess_meta['vocab_after_filter'],
        'names_registered': preprocess_meta['names_count'],
        'feature_nonzero_ratio': nonzero_ratio,
        'chapter_token_stats': _describe_array(token_counts),
        'chapter_unique_stats': _describe_array(unique_counts),
        'chapter_para_stats': _describe_array(para_counts),
        'front_token_stats': _describe_array(front_tokens),
        'back_token_stats': _describe_array(back_tokens),
        'front_unique_stats': _describe_array(front_unique),
        'back_unique_stats': _describe_array(back_unique),
        'front_para_stats': _describe_array(front_para),
        'back_para_stats': _describe_array(back_para),
        'top_words': top_words,
        'chapter_ids': chapter_ids,
        'token_counts': token_counts,
        'unique_counts': unique_counts,
        'paragraph_counts': para_counts,
    }


def plot_chapter_token_trend(desc_stats, n_front=N_FRONT):
    """各章有效词数随回目变化折线图。"""
    chapter_ids = desc_stats['chapter_ids']
    token_counts = desc_stats['token_counts']

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(chapter_ids, token_counts, color='steelblue', linewidth=1.2, marker='o', markersize=2.5)
    ax.axvline(n_front + 0.5, color='crimson', linestyle='--', linewidth=1.0, label='第 80/81 回分界')
    ax.set_xlabel('回目序号')
    ax.set_ylabel('有效词数（去停用词、去单字）')
    ax.set_title('各章有效词数变化趋势')
    ax.legend()
    return _save_fig('desc_chapter_token_trend.png')


def plot_chapter_unique_trend(desc_stats, n_front=N_FRONT):
    """各章不同词数（类型数）随回目变化折线图。"""
    chapter_ids = desc_stats['chapter_ids']
    unique_counts = desc_stats['unique_counts']

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(chapter_ids, unique_counts, color='darkorange', linewidth=1.2, marker='o', markersize=2.5)
    ax.axvline(n_front + 0.5, color='crimson', linestyle='--', linewidth=1.0, label='第 80/81 回分界')
    ax.set_xlabel('回目序号')
    ax.set_ylabel('不同词数（类型数）')
    ax.set_title('各章词汇丰富度（类型数）变化趋势')
    ax.legend()
    return _save_fig('desc_chapter_unique_trend.png')


def plot_top_words_bar(desc_stats):
    """全书 Top 高频非人名词条形图。"""
    words = [item['word'] for item in desc_stats['top_words']]
    freqs = [item['freq'] for item in desc_stats['top_words']]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(words))
    ax.barh(y_pos, freqs, color='seagreen')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('全书出现频次')
    ax.set_title(f'预处理特征词 Top {len(words)}（已剔除人名）')
    return _save_fig('desc_top_words.png')


def plot_chapter_stats_boxplot(desc_stats, n_front=N_FRONT):
    """前 80 回 vs 后 40 回：词数、类型数、段落数箱线图。"""
    metrics = [
        ('token_counts', '有效词数'),
        ('unique_counts', '不同词数'),
        ('paragraph_counts', '段落数'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, label) in zip(axes, metrics):
        values = desc_stats[key]
        front, back = _group_split(values, n_front)
        ax.boxplot([front, back], tick_labels=['前 80 回', '后 40 回'])
        jitter_front = np.random.default_rng(0).uniform(-0.08, 0.08, len(front))
        jitter_back = np.random.default_rng(1).uniform(-0.08, 0.08, len(back))
        ax.scatter(1 + jitter_front, front, alpha=0.35, s=18, color='C0')
        ax.scatter(2 + jitter_back, back, alpha=0.35, s=18, color='C1')
        ax.set_title(label)
    fig.suptitle('章节基本指标分组对比（描述性统计）')
    return _save_fig('desc_chapter_stats_boxplot.png')


def plot_group_mean_comparison(desc_stats):
    """前 80 回 vs 后 40 回：关键指标均值对比条形图。"""
    labels = ['有效词数', '不同词数', '段落数']
    front_means = [
        desc_stats['front_token_stats']['mean'],
        desc_stats['front_unique_stats']['mean'],
        desc_stats['front_para_stats']['mean'],
    ]
    back_means = [
        desc_stats['back_token_stats']['mean'],
        desc_stats['back_unique_stats']['mean'],
        desc_stats['back_para_stats']['mean'],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, front_means, width, label='前 80 回', color='C0')
    ax.bar(x + width / 2, back_means, width, label='后 40 回', color='C1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('章节均值')
    ax.set_title('前后两部分章节指标均值对比')
    ax.legend()
    return _save_fig('desc_group_mean_comparison.png')


def plot_feature_sparsity_heatmap(features, word_list, max_words=40):
    """特征矩阵稀疏模式热力图（展示前若干高频词在各章的使用情况）。"""
    n_words = min(max_words, features.shape[1])
    data = features[:, :n_words]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('特征词（按高频排序）')
    ax.set_ylabel('回目序号（1–120）')
    ax.set_title(f'特征矩阵稀疏热力图（Top {n_words} 词 × 120 回）')
    tick_step = max(1, n_words // 10)
    ax.set_xticks(np.arange(0, n_words, tick_step))
    ax.set_xticklabels([word_list[i] for i in range(0, n_words, tick_step)], rotation=45, ha='right')
    fig.colorbar(im, ax=ax, label='归一化词频')
    return _save_fig('desc_feature_sparsity_heatmap.png')


def run_preprocessing_visualization(desc_stats, features, word_list):
    """生成全部预处理描述性统计图表。"""
    return [
        plot_chapter_token_trend(desc_stats),
        plot_chapter_unique_trend(desc_stats),
        plot_top_words_bar(desc_stats),
        plot_chapter_stats_boxplot(desc_stats),
        plot_group_mean_comparison(desc_stats),
        plot_feature_sparsity_heatmap(features, word_list),
    ]


def format_descriptive_report(desc_stats):
    """格式化描述性统计文本报告。"""
    sep = '=' * 60
    lines = []

    def add(text=''):
        lines.append(text) 

    def fmt_stats(title, stats_dict):
        add(f'  {title}:')
        add(f'    n={stats_dict["n"]}, 均值={stats_dict["mean"]:.2f}, '
            f'标准差={stats_dict["std"]:.2f}, 中位数={stats_dict["median"]:.2f}')
        add(f'    最小值={stats_dict["min"]:.0f}, Q1={stats_dict["q1"]:.2f}, '
            f'Q3={stats_dict["q3"]:.2f}, 最大值={stats_dict["max"]:.0f}')

    add(sep)
    add('描述性统计（预处理后）')
    add(sep)
    add(f'  章节总数: {desc_stats["n_chapters"]}')
    add(f'  全书有效词次总数: {desc_stats["total_tokens"]}')
    add(f'  全书不同词数（词汇量）: {desc_stats["book_vocab"]}')
    add(f'  去人名前不同词数: {desc_stats["vocab_before_filter"]}')
    add(f'  去人名后不同词数: {desc_stats["vocab_after_filter"]}')
    add(f'  特征维度 V: {desc_stats["n_features"]}')
    add(f'  特征矩阵非零占比: {desc_stats["feature_nonzero_ratio"]:.4f}')
    add(f'  注册人物别名数: {desc_stats["names_registered"]}')

    add('\n  [各章有效词数 — 全书]')
    fmt_stats('有效词数', desc_stats['chapter_token_stats'])
    add('\n  [各章不同词数 — 全书]')
    fmt_stats('不同词数', desc_stats['chapter_unique_stats'])
    add('\n  [各章段落数 — 全书]')
    fmt_stats('段落数', desc_stats['chapter_para_stats'])

    add('\n  [前 80 回 vs 后 40 回 — 有效词数]')
    fmt_stats('前 80 回', desc_stats['front_token_stats'])
    fmt_stats('后 40 回', desc_stats['back_token_stats'])
    add('\n  [前 80 回 vs 后 40 回 — 不同词数]')
    fmt_stats('前 80 回', desc_stats['front_unique_stats'])
    fmt_stats('后 40 回', desc_stats['back_unique_stats'])

    add('\n  Top 10 高频非人名词:')
    add(f'  {"词语":<12} {"全书频次":>10}')
    for item in desc_stats['top_words'][:10]:
        add(f'  {item["word"]:<12} {item["freq"]:>10}')

    return '\n'.join(lines)


# =============================================================================
# 第三部分：全特征非参数检验（核心推断）
# =============================================================================
def rank_biserial(u_stat, n1, n2):
    """Mann-Whitney U 检验的 rank-biserial 效应量。"""
    return 1 - (2 * u_stat) / (n1 * n2)


def mann_whitney_two_sample(x, y):
    """两独立样本 Mann-Whitney U 检验。"""
    res = stats.mannwhitneyu(x, y, alternative='two-sided')
    r = rank_biserial(res.statistic, len(x), len(y))
    return {'U': res.statistic, 'p': res.pvalue, 'rank_biserial_r': r}


def ks_two_sample(x, y):
    """两独立样本 Kolmogorov-Smirnov 检验。"""
    res = stats.ks_2samp(x, y)
    return {'statistic': res.statistic, 'p': res.pvalue}


def feature_wise_tests(features, word_list, n_front=N_FRONT, alpha=ALPHA):
    """
    对全部 V 个词频特征逐一做 Mann-Whitney U 检验，
    并进行 Bonferroni 与 FDR (BH) 多重比较校正。
    另对 Top 代表词补充 KS 检验，供可视化解读。
    """
    x_front = features[:n_front]
    x_back = features[n_front:]
    n_features = features.shape[1]
    u_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    effect_sizes = np.zeros(n_features)

    for j in tqdm(range(n_features), desc='逐词 Mann-Whitney'):
        res = stats.mannwhitneyu(x_front[:, j], x_back[:, j], alternative='two-sided')
        u_stats[j] = res.statistic
        p_values[j] = res.pvalue
        effect_sizes[j] = rank_biserial(res.statistic, n_front, len(x_back))

    bonferroni_p = np.minimum(p_values * n_features, 1.0)
    fdr_p = stats.false_discovery_control(p_values, method='bh')

    sig_raw = int(np.sum(p_values < alpha))
    sig_bonf = int(np.sum(bonferroni_p < alpha))
    sig_fdr = int(np.sum(fdr_p < alpha))

    # Top 差异词（按原始 p 值）
    order = np.argsort(p_values)
    top_records = []
    for idx in order[:15]:
        top_records.append({
            'word': word_list[idx],
            'U': u_stats[idx],
            'p': p_values[idx],
            'bonferroni_p': bonferroni_p[idx],
            'fdr_p': fdr_p[idx],
            'rank_biserial_r': effect_sizes[idx],
        })

    # FDR 显著词
    fdr_sig_indices = np.where(fdr_p < alpha)[0]
    fdr_sig_sorted = fdr_sig_indices[np.argsort(p_values[fdr_sig_indices])]
    top_fdr_words = [
        {
            'word': word_list[idx],
            'p': p_values[idx],
            'fdr_p': fdr_p[idx],
            'rank_biserial_r': effect_sizes[idx],
        }
        for idx in fdr_sig_sorted[:15]
    ]

    # 代表词：优先从 FDR 显著词中选取，不足则回退到原始 p 值 Top 词
    rep_source = top_fdr_words if top_fdr_words else top_records
    representative = []
    seen_words = set()
    for rec in rep_source:
        word = rec['word']
        if word in seen_words:
            continue
        seen_words.add(word)
        idx = word_list.index(word)
        front_vals = x_front[:, idx]
        back_vals = x_back[:, idx]
        mw = mann_whitney_two_sample(front_vals, back_vals)
        ks = ks_two_sample(front_vals, back_vals)
        representative.append({
            'word': word,
            'index': idx,
            'U': mw['U'],
            'p': rec.get('p', p_values[idx]),
            'bonferroni_p': bonferroni_p[idx],
            'fdr_p': rec.get('fdr_p', fdr_p[idx]),
            'rank_biserial_r': mw['rank_biserial_r'],
            'ks_D': ks['statistic'],
            'ks_p': ks['p'],
        })
        if len(representative) >= N_REP_FEATURES:
            break

    return {
        'p_values': p_values,
        'bonferroni_p': bonferroni_p,
        'fdr_p': fdr_p,
        'effect_sizes': effect_sizes,
        'sig_raw': sig_raw,
        'sig_bonferroni': sig_bonf,
        'sig_fdr': sig_fdr,
        'top_records': top_records,
        'top_fdr_words': top_fdr_words,
        'representative': representative,
    }


# =============================================================================
# 第四部分：PCA 降维 + PC1 补充检验
# =============================================================================
def run_pca(features, n_components=N_COMPONENTS, scree_components=SCREE_COMPONENTS):
    """
    对特征矩阵做 PCA 降维。
    返回：前 n_components 维得分、所选成分解释率、完整碎石图数据。
    """
    n_scree = min(scree_components, features.shape[0], features.shape[1])
    pca_full = decomposition.PCA(n_components=n_scree)
    pca_full.fit(features)
    scores = pca_full.transform(features)[:, :n_components]
    return scores, pca_full.explained_variance_ratio_[:n_components], pca_full.explained_variance_ratio_


def _pc1_mean_diff(pc1, labels):
    """PC1 上两组样本均值的绝对差，用作置换检验统计量。"""
    front_mean = pc1[labels == 0].mean()
    back_mean = pc1[labels == 1].mean()
    return abs(front_mean - back_mean)


def permutation_test_pc1(scores, n_front=N_FRONT, n_perm=N_PERM, seed=42):
    """
    PC1 置换检验：在原标签下计算两组 PC1 均值差，
    重复打乱组标签，构造零分布并计算 p 值。
    """
    rng = np.random.default_rng(seed)
    pc1 = scores[:, 0]
    labels = np.array([0] * n_front + [1] * (len(scores) - n_front))
    t_obs = _pc1_mean_diff(pc1, labels)
    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        if _pc1_mean_diff(pc1, perm_labels) >= t_obs:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {'T_obs': t_obs, 'p': p_value, 'n_perm': n_perm}


def pc1_tests(scores, n_front=N_FRONT):
    """
    对第一主成分（PC1）做三项非参数检验：
    Mann-Whitney U、Kolmogorov-Smirnov、置换检验。
    PC1 解释方差最大，作为 PCA 方向的代表性检验。
    """
    pc1_front = scores[:n_front, 0]
    pc1_back = scores[n_front:, 0]
    return {
        'mann_whitney': mann_whitney_two_sample(pc1_front, pc1_back),
        'ks': ks_two_sample(pc1_front, pc1_back),
        'permutation': permutation_test_pc1(scores, n_front=n_front),
    }


# =============================================================================
# 第五部分：推断性分析可视化
# =============================================================================

# --- 5.1 全特征检验相关图表 ---

def plot_top_words(feature_results):
    """差异最显著词语的 -log10(p) 条形图。"""
    words_data = feature_results['top_fdr_words']
    if not words_data:
        words_data = feature_results['top_records'][:15]

    words = [d['word'] for d in words_data]
    neg_log_p = [-np.log10(max(d['p'], 1e-300)) for d in words_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(words))
    ax.barh(y_pos, neg_log_p, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel(r'$-\log_{10}(p)$')
    ax.set_title('差异最显著的前 15 个词语（Mann-Whitney）')
    ax.axvline(-np.log10(ALPHA), color='red', linestyle='--', linewidth=0.8, label=f'α={ALPHA}')
    ax.legend()
    return _save_fig('top_diff_words.png')


def plot_feature_boxplots(features, representative, n_front=N_FRONT):
    """代表词的归一化词频箱线图（前 80 回 vs 后 40 回）。"""
    n_rep = len(representative)
    n_cols = min(3, n_rep)
    n_rows = (n_rep + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, rec in zip(axes, representative):
        idx = rec['index']
        data_front = features[:n_front, idx]
        data_back = features[n_front:, idx]
        ax.boxplot([data_front, data_back], tick_labels=['前 80 回', '后 40 回'])
        jitter_front = np.random.default_rng(0).uniform(-0.08, 0.08, len(data_front))
        jitter_back = np.random.default_rng(1).uniform(-0.08, 0.08, len(data_back))
        ax.scatter(1 + jitter_front, data_front, alpha=0.4, s=15, color='C0')
        ax.scatter(2 + jitter_back, data_back, alpha=0.4, s=15, color='C1')
        ax.set_title(f'「{rec["word"]}」  p={rec["p"]:.2e}')
        ax.set_ylabel('归一化词频')

    for ax in axes[n_rep:]:
        ax.set_visible(False)

    fig.suptitle('代表性差异词语 — 分组箱线图')
    return _save_fig('feature_boxplot.png')


def plot_feature_density(features, representative, n_front=N_FRONT):
    """代表词的分布密度对比图（直方图 + KDE）。"""
    n_rep = len(representative)
    n_cols = min(3, n_rep)
    n_rows = (n_rep + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, rec in zip(axes, representative):
        idx = rec['index']
        data_front = features[:n_front, idx]
        data_back = features[n_front:, idx]
        ax.hist(data_front, bins=15, density=True, alpha=0.5, label='前 80 回')
        ax.hist(data_back, bins=15, density=True, alpha=0.5, label='后 40 回')
        try:
            x_front = np.linspace(data_front.min(), data_front.max(), 200)
            x_back = np.linspace(data_back.min(), data_back.max(), 200)
            ax.plot(x_front, stats.gaussian_kde(data_front)(x_front))
            ax.plot(x_back, stats.gaussian_kde(data_back)(x_back))
        except np.linalg.LinAlgError:
            pass
        ax.set_title(f'「{rec["word"]}」  KS p={rec["ks_p"]:.2e}')
        ax.set_xlabel('归一化词频')
        ax.legend(fontsize=8)

    for ax in axes[n_rep:]:
        ax.set_visible(False)

    fig.suptitle('代表性差异词语 — 分布密度对比')
    return _save_fig('feature_density.png')


def run_feature_visualization(features, feature_results):
    """生成全特征检验相关的全部图表。"""
    return [
        plot_top_words(feature_results),
        plot_feature_boxplots(features, feature_results['representative']),
        plot_feature_density(features, feature_results['representative']),
    ]


# --- 5.2 PCA 探索性可视化图表 ---

def plot_pca_scatter(scores, n_front=N_FRONT):
    """PCA 三维散点图：前 80 回 vs 后 40 回。"""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        scores[:n_front, 0], scores[:n_front, 1], scores[:n_front, 2],
        alpha=0.7, label='前 80 回', s=30,
    )
    ax.scatter(
        scores[n_front:, 0], scores[n_front:, 1], scores[n_front:, 2],
        alpha=0.7, label='后 40 回', s=30,
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA 三维散点图：前 80 回 vs 后 40 回')
    ax.legend()
    return _save_fig('pca_scatter.png')


def plot_scree(explained_ratio_full):
    """PCA 碎石图：展示各主成分方差解释率。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(explained_ratio_full) + 1)
    ax.bar(x, explained_ratio_full * 100, alpha=0.7, label='单个主成分')
    ax.plot(x, np.cumsum(explained_ratio_full) * 100, 'o-', color='C1', label='累计解释率')
    ax.set_xlabel('主成分')
    ax.set_ylabel('方差解释率 (%)')
    ax.set_title('PCA 碎石图')
    ax.legend()
    return _save_fig('pca_scree.png')


def plot_pc1_boxplot(scores, pc1_results, n_front=N_FRONT):
    """PC1 分组箱线图（含散点抖动）。"""
    mw_p = pc1_results['mann_whitney']['p']
    fig, ax = plt.subplots(figsize=(6, 4))
    data_front = scores[:n_front, 0]
    data_back = scores[n_front:, 0]
    ax.boxplot([data_front, data_back], tick_labels=['前 80 回', '后 40 回'])
    jitter_front = np.random.default_rng(0).uniform(-0.08, 0.08, len(data_front))
    jitter_back = np.random.default_rng(1).uniform(-0.08, 0.08, len(data_back))
    ax.scatter(1 + jitter_front, data_front, alpha=0.4, s=20, color='C0')
    ax.scatter(2 + jitter_back, data_back, alpha=0.4, s=20, color='C1')
    ax.set_title(f'PC1 分组箱线图  Mann-Whitney p={mw_p:.2e}')
    ax.set_ylabel('PC1 得分')
    return _save_fig('pc1_boxplot.png')


def plot_pc1_density(scores, pc1_results, n_front=N_FRONT):
    """PC1 分组分布密度对比图。"""
    ks_p = pc1_results['ks']['p']
    fig, ax = plt.subplots(figsize=(6, 4))
    data_front = scores[:n_front, 0]
    data_back = scores[n_front:, 0]
    ax.hist(data_front, bins=20, density=True, alpha=0.5, label='前 80 回')
    ax.hist(data_back, bins=20, density=True, alpha=0.5, label='后 40 回')
    try:
        x_front = np.linspace(data_front.min(), data_front.max(), 200)
        x_back = np.linspace(data_back.min(), data_back.max(), 200)
        ax.plot(x_front, stats.gaussian_kde(data_front)(x_front), label='前 80 回 KDE')
        ax.plot(x_back, stats.gaussian_kde(data_back)(x_back), label='后 40 回 KDE')
    except np.linalg.LinAlgError:
        pass
    ax.set_title(f'PC1 分布对比  KS p={ks_p:.2e}')
    ax.set_xlabel('PC1 得分')
    ax.legend(fontsize=8)
    return _save_fig('pc1_density.png')


def run_pca_analysis(features):
    """
    PCA 降维、PC1 三项检验，并生成相关图表。
    返回：scores, pca_ratio, pc1_results, 图表路径列表。
    """
    scores, pca_ratio, explained_full = run_pca(features)
    pc1_results = pc1_tests(scores)
    paths = [
        plot_pca_scatter(scores),
        plot_scree(explained_full),
        plot_pc1_boxplot(scores, pc1_results),
        plot_pc1_density(scores, pc1_results),
    ]
    return scores, pca_ratio, pc1_results, paths


# =============================================================================
# 第六部分：统计报告
# =============================================================================
def format_report(desc_stats, feature_results, pca_ratio, pc1_results, n_chapters, n_features,
                  saved_paths, report_path=REPORT_PATH):
    """组装完整文本报告（描述性统计在前，推断结果在后）。"""
    sep = '=' * 60
    pc_labels = ', '.join(f'PC{i + 1}' for i in range(len(pca_ratio)))
    lines = []

    def add(text=''):
        lines.append(text)

    add(format_descriptive_report(desc_stats))

    # ----- 数据概况 -----
    add(f'\n{sep}')
    add('推断分析数据概况')
    add(sep)
    add(f'  章节数: {n_chapters}')
    add(f'  特征维度 V: {n_features}')
    add(f'  前 80 回样本量: {N_FRONT}')
    add(f'  后 40 回样本量: {n_chapters - N_FRONT}')

    # ----- 全特征检验（核心推断，优先呈现）-----
    add(f'\n{sep}')
    add(f'一、全特征逐词检验（{n_features} 特征）')
    add(sep)
    add('  [Mann-Whitney U 检验 + 多重比较校正]')
    add(f'  原始 p < {ALPHA} 的特征数: {feature_results["sig_raw"]}')
    add(f'  Bonferroni 校正后显著: {feature_results["sig_bonferroni"]}')
    add(f'  FDR (BH) 校正后显著: {feature_results["sig_fdr"]}')
    add('\n  Top 10 差异词（按原始 p 值排序）:')
    add(f'  {"词语":<12} {"U":>10} {"p":>12} {"Bonf.p":>12} {"FDR.p":>12} {"r":>8}')
    for rec in feature_results['top_records'][:10]:
        add(f'  {rec["word"]:<12} {rec["U"]:>10.1f} {rec["p"]:>12.4e} '
            f'{rec["bonferroni_p"]:>12.4e} {rec["fdr_p"]:>12.4e} {rec["rank_biserial_r"]:>8.4f}')

    add(f'\n  代表性特征补充检验（Top {len(feature_results["representative"])}，Mann-Whitney + KS）:')
    add(f'  {"词语":<12} {"MW.p":>12} {"KS D":>8} {"KS p":>12} {"r":>8}')
    for rec in feature_results['representative']:
        add(f'  {rec["word"]:<12} {rec["p"]:>12.4e} {rec["ks_D"]:>8.4f} '
            f'{rec["ks_p"]:>12.4e} {rec["rank_biserial_r"]:>8.4f}')

    # ----- PCA 降维 + PC1 补充检验 -----
    add(f'\n{sep}')
    add('二、PCA 降维与 PC1 补充检验')
    add(sep)
    add(f'  主成分数: {len(pca_ratio)}')
    for i, ratio in enumerate(pca_ratio, 1):
        add(f'  PC{i} 解释方差比: {ratio:.4f} ({ratio * 100:.2f}%)')
    add(f'  {pc_labels} 累计解释率: {pca_ratio.sum():.4f} ({pca_ratio.sum() * 100:.2f}%)')
    add('  说明: PC1 解释方差最大，对其做 Mann-Whitney / KS / 置换检验作为 PCA 方向的补充验证。')

    mw = pc1_results['mann_whitney']
    ks = pc1_results['ks']
    perm = pc1_results['permutation']
    add('\n  [PC1 — Mann-Whitney U 检验]')
    add(f'    U={mw["U"]:.1f}, p={mw["p"]:.4e}, rank-biserial r={mw["rank_biserial_r"]:.4f}')
    add('  [PC1 — Kolmogorov-Smirnov 两样本检验]')
    add(f'    D={ks["statistic"]:.4f}, p={ks["p"]:.4e}')
    add('  [PC1 — 置换检验（两组 PC1 均值差的绝对值）]')
    add(f'    T_obs={perm["T_obs"]:.6f}, p_perm={perm["p"]:.4e}, 置换次数={perm["n_perm"]}')

    # ----- 综合结论 -----
    add(f'\n{sep}')
    add('结论')
    add(sep)
    evidence = []
    if feature_results['sig_raw'] > 0:
        evidence.append(f'原始检验有 {feature_results["sig_raw"]} 个词语 p < {ALPHA}')
    if feature_results['sig_bonferroni'] > 0:
        evidence.append(f'Bonferroni 校正后有 {feature_results["sig_bonferroni"]} 个词语显著')
    if feature_results['sig_fdr'] > 0:
        evidence.append(f'FDR 校正后有 {feature_results["sig_fdr"]} 个词语显著差异')
    rep_sig_ks = sum(1 for rec in feature_results['representative'] if rec['ks_p'] < ALPHA)
    if rep_sig_ks > 0:
        evidence.append(f'代表性特征中有 {rep_sig_ks} 个 KS 检验显著')
    if mw['p'] < ALPHA:
        evidence.append('PC1 Mann-Whitney 检验显著')
    if ks['p'] < ALPHA:
        evidence.append('PC1 KS 检验显著')
    if perm['p'] < ALPHA:
        evidence.append('PC1 置换检验显著')

    if evidence:
        add(f'  在 α={ALPHA} 下，全特征非参数检验提示前 80 回与后 40 回存在用词差异：')
        for item in evidence:
            add(f'    - {item}')
        add('  综合而言，用词风格在前后两部分之间并非完全一致，与「后四十回作者存疑」的假设相容。')
    else:
        add(f'  在 α={ALPHA} 下，各项检验均未发现显著差异，不能拒绝「前后回风格相同」的假设。')

    add(f'\n{sep}')
    add('图表已保存')
    add(sep)
    for path in saved_paths:
        add(f'  {path}')
    add(f'\n{sep}')
    add('检验报告已保存')
    add(sep)
    add(f'  {report_path}')

    return '\n'.join(lines)


def save_report(report_text, report_path=REPORT_PATH):
    """将报告写入 txt 文件。"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    return report_path


def print_report(desc_stats, feature_results, pca_ratio, pc1_results, n_chapters, n_features,
                 saved_paths, report_path=REPORT_PATH):
    """打印并保存统计报告。"""
    report_text = format_report(
        desc_stats, feature_results, pca_ratio, pc1_results,
        n_chapters, n_features, saved_paths, report_path=report_path,
    )
    print(report_text, flush=True)
    save_report(report_text, report_path=report_path)


# =============================================================================
# 主流程
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 步骤 0：预处理，构建特征矩阵 ----
    print('【步骤 0】文本预处理与特征构建...')
    features, word_list, chapter_ids, preprocess_meta = prepare_features()

    # ---- 步骤 1：描述性统计与预处理可视化 ----
    print('【步骤 1】描述性统计与预处理可视化...')
    desc_stats = compute_descriptive_stats(preprocess_meta, features, word_list, chapter_ids)
    desc_plot_paths = run_preprocessing_visualization(desc_stats, features, word_list)
    print(format_descriptive_report(desc_stats))

    # ---- 步骤 2：全特征非参数检验（核心推断）----
    print('【步骤 2】全特征 Mann-Whitney 检验 + 多重比较校正...')
    feature_results = feature_wise_tests(features, word_list)

    print('【步骤 2】生成检验相关图表...')
    feature_plot_paths = run_feature_visualization(features, feature_results)

    # ---- 步骤 3：PCA 降维 + PC1 补充检验 ----
    print('【步骤 3】PCA 降维、PC1 三项检验与可视化...')
    _, pca_ratio, pc1_results, pca_plot_paths = run_pca_analysis(features)

    saved_paths = desc_plot_paths + feature_plot_paths + pca_plot_paths

    # ---- 步骤 4：输出报告 ----
    print('【步骤 4】输出统计报告...')
    print_report(
        desc_stats, feature_results, pca_ratio, pc1_results,
        n_chapters=len(chapter_ids), n_features=len(word_list), saved_paths=saved_paths,
    )

    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()
