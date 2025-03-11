import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import entropy

# 设定参数
num_total_students = 50  # 总学生数
num_total_problems = 50  # 总问题数
num_selected_students = 10  # 选取 10 个学生用于二部图
num_selected_problems = 10  # 选取 10 个问题用于二部图

# 生成完整数据（所有学生答所有问题）
user_problem_pairs = [(user, problem) for user in range(1, num_total_students + 1) for problem in range(1, num_total_problems + 1)]
df = pd.DataFrame(user_problem_pairs, columns=['user_id', 'problem_id'])

# 生成时间戳（保证顺序性）
time_range = pd.date_range(start="2023-01-01", periods=len(df), freq='h')
df['timestamp'] = np.random.choice(time_range, size=len(df), replace=False)

# 生成正确率（学生是否正确回答题目），基于 Rasch 模型
theta = np.random.normal(0, 1, num_total_students)  # 学生能力参数
beta = np.random.normal(0, 1, num_total_problems)  # 题目难度参数

df['true_theta'] = df['user_id'].map(lambda x: theta[x - 1])
df['true_beta'] = df['problem_id'].map(lambda x: beta[x - 1])
df['prob_correct'] = np.exp(df['true_theta'] - df['true_beta']) / (1 + np.exp(df['true_theta'] - df['true_beta']))
df['correct'] = np.random.binomial(1, df['prob_correct'])  # 生成 0/1 回答正确与否

# 按时间排序
df.sort_values('timestamp', inplace=True)

# **最大似然估计 (MLE) 计算 Rasch 模型参数**
def rasch_log_likelihood(params, df):
    """计算 Rasch 似然函数，参数包括学生能力 (theta) 和 题目难度 (beta)"""
    theta_est = params[:num_total_students]
    beta_est = params[num_total_students:]
    
    log_likelihood = 0
    for _, row in df.iterrows():
        user = int(row['user_id']) - 1
        problem = int(row['problem_id']) - 1
        p = np.exp(theta_est[user] - beta_est[problem]) / (1 + np.exp(theta_est[user] - beta_est[problem]))
        log_likelihood += row['correct'] * np.log(p + 1e-10) + (1 - row['correct']) * np.log(1 - p + 1e-10)
    
    return -log_likelihood  # 负对数似然（用于最优化）

# 初始化参数
initial_params = np.random.normal(0, 1, num_total_students + num_total_problems)
result = minimize(rasch_log_likelihood, initial_params, args=(df,), method='BFGS')

# 解析优化结果
theta_estimated = result.x[:num_total_students]
beta_estimated = result.x[num_total_students:]

# 添加 Rasch 估计值
df['theta_est'] = df['user_id'].map(lambda x: theta_estimated[x - 1])
df['beta_est'] = df['problem_id'].map(lambda x: beta_estimated[x - 1])

# 计算滚动能力估计值
df['theta_rolling'] = df.groupby('user_id')['theta_est'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

# 计算 KL 散度（基于能力估计值变化）
def kl_divergence(p, q):
    p = np.array(p) + 1e-10  # 避免 log(0)
    q = np.array(q) + 1e-10
    return entropy(p, q)

df['kl_divergence'] = 0.0

for user_id in df['user_id'].unique():
    user_df = df[df['user_id'] == user_id]
    prev_probs = None
    kl_values = []
    
    for i in range(len(user_df)):
        current_probs = np.histogram(user_df.iloc[max(0, i - 5):i + 1]['theta_est'], bins=10, density=True)[0]
        if prev_probs is not None:
            kl_values.append(kl_divergence(prev_probs, current_probs))
        else:
            kl_values.append(0)  # 初始时 KL 散度设为 0
        prev_probs = current_probs
    
    df.loc[df['user_id'] == user_id, 'kl_divergence'] = kl_values

# 变点检测（使用 KL 散度 > 0.3 作为阈值）
df['change_point'] = (df['kl_divergence'] > 0.3).astype(int)

# **绘制滚动能力变化曲线**
selected_students = np.random.choice(df['user_id'].unique(), 5, replace=False)
df_sampled = df[df['user_id'].isin(selected_students)]

plt.figure(figsize=(10, 5))
for user_id, user_df in df_sampled.groupby('user_id'):
    plt.plot(user_df['timestamp'], user_df['theta_rolling'], marker='o', linestyle='-', alpha=0.7, label=f"User {int(user_id)}")  

# 标记变点（使用 KL 散度）
plt.scatter(df_sampled['timestamp'][df_sampled['change_point'] == 1], df_sampled['theta_rolling'][df_sampled['change_point'] == 1], 
            color='red', label="Change Points (KL Divergence)")
plt.xlabel("Timestamp")
plt.ylabel("Rolling Ability Estimate")
plt.title("Change Point Detection in Ability Estimates (KL Divergence) - 5 Sampled Users")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# **二部图分析**
selected_students = np.random.choice(df['user_id'].unique(), num_selected_students, replace=False)
selected_problems = np.random.choice(df['problem_id'].unique(), num_selected_problems, replace=False)

df_bipartite = pd.DataFrame([(user, problem) for user in selected_students for problem in selected_problems], columns=['user_id', 'problem_id'])
df_bipartite = df_bipartite.merge(df[['user_id', 'problem_id', 'theta_est']], on=['user_id', 'problem_id'], how='left')

# **构建二部图**
graph = nx.Graph()

for _, row in df_bipartite.iterrows():
    graph.add_edge(f"User {int(row['user_id'])}", f"Problem {int(row['problem_id'])}", weight=row['theta_est'])

# 绘制二部图
plt.figure(figsize=(12, 7))
nx.draw(graph, with_labels=True, node_size=500, font_size=8, edge_color='gray', alpha=0.5)
plt.title("Bipartite Graph with Estimated Ability (Rasch Model)")
plt.show()
