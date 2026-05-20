import streamlit as st
import pandas as pd
from openai import OpenAI
import random # ✨ 新增：引入随机数库，用于实现探索因子

# ==========================================
# 1. 初始化设置与界面
# ==========================================
st.set_page_config(page_title="济时饭 Agent", page_icon="🍽️")
st.title("🍽️ 济时饭 —— 校园饮食推荐 Agent")
st.write("基于同济大学真实距离矩阵与大模型的个性化推荐系统")

# 侧边栏大模型配置
st.sidebar.header("⚙️ 智能体后台配置")
api_key = st.sidebar.text_input("输入你的 API Key:", type="password", help="请输入你的大模型 API 密钥")
base_url = st.sidebar.text_input("输入 Base URL:", value="https://api.deepseek.com/v1")
model_name = st.sidebar.text_input("模型名称:", value="deepseek-chat")

st.divider()
st.subheader("🎒 告诉老学长你现在的情况：")

current_location = st.selectbox(
    "1. 你当前在哪栋楼或哪个区域？", 
    ["南校区", "衷和楼", "瑞安楼", "南楼", "德文图书馆", "主图", "体育馆", "北楼", "大礼堂"]
)

budget = st.slider("2. 你的预算是多少元？", min_value=5, max_value=50, value=18)

time_left = st.radio(
    "3. 你现在的吃饭时间紧不紧张？", 
    ["赶时间（<20分钟）", "正常（30-40分钟）", "时间很充裕（慢慢吃）"]
)

taste_preference = st.multiselect(
    "4. 今天想吃什么口味？(可选多个)", 
    ["咸鲜", "麻辣", "咸甜", "清淡", "汤面", "酸甜", "酱香", "咸香", "酸爽", "微辣", "鲜香", "酸辣"]
)

st.divider()

# ==========================================
# 2. 辅助函数：标准化食堂名称
# ==========================================
def clean_canteen_name(name):
    return str(name).replace("餐厅", "").strip()

# ==========================================
# 3. 核心算法：基于物理距离矩阵与探索因子的评分系统
# ==========================================
def run_recommendation(df_meals, df_distance, user_loc, user_budget, user_time, user_tastes):
    df = df_meals.copy()
    
    # 【硬性约束】价格过滤
    df = df[df["Price"] <= user_budget]
    if df.empty:
        return df

    # 预处理距离矩阵
    df_distance.columns = df_distance.columns.str.strip()
    rename_dict = {col: clean_canteen_name(col) for col in df_distance.columns if col != 'Location'}
    df_distance = df_distance.rename(columns=rename_dict)
    df_distance['Location'] = df_distance['Location'].str.strip()
    df_distance = df_distance.set_index('Location')

    # 初始化基础数据
    df["Distance_Meters"] = 9999
    df["Score"] = 50.0

    for index, row in df.iterrows():
        score = 50.0
        reasons = []
        
        # 🟢 维度一：物理距离精准得分
        meal_canteen_clean = clean_canteen_name(row["Canteen"])
        if user_loc in df_distance.index and meal_canteen_clean in df_distance.columns:
            dist_str = str(df_distance.loc[user_loc, meal_canteen_clean]).lower().replace('m', '').strip()
            try:
                dist_meters = float(dist_str)
                df.at[index, "Distance_Meters"] = int(dist_meters)
                dist_score = max(0.0, 60.0 - (dist_meters / 15.0))
                score += dist_score
                reasons.append(f"距离{int(dist_meters)}米(+{dist_score:.1f}分)")
            except:
                reasons.append("距离未知")
        else:
            reasons.append("未匹配到路线")
            
        # 🔵 维度二：时间约束打分
        if user_time == "赶时间（<20分钟）" and row["Speed_Min"] > 4:
            score -= 20
            reasons.append(f"出餐慢(-20分)")
            
        # 🟡 维度三：口味偏好打分
        if user_tastes:
            match_count = sum([1 for taste in user_tastes if taste in str(row["Taste_Tags"])])
            if match_count > 0:
                taste_bonus = match_count * 10
                score += taste_bonus
                reasons.append(f"口味匹配(+{taste_bonus}分)")
            else:
                score -= 10
                reasons.append("口味不符(-10分)")

        # 🔮 【核心高亮】维度四：AI 探索因子（Exploration Factor）
        # 产生一个 -2.0 到 +2.0 之间的随机波动，模拟推荐系统的新鲜感避免死板
        exploration_noise = random.uniform(-2.0, 2.0)
        score += exploration_noise
        reasons.append(f"趣味探索因子({'+' if exploration_noise >= 0 else ''}{exploration_noise:.1f}分)")

        df.at[index, "Score"] = round(score, 1)
        df.at[index, "Reason"] = " | ".join(reasons)
        
    return df.sort_values(by=["Score", "Distance_Meters"], ascending=[False, True])

# ==========================================
# 4. 精简对比版 AI 推荐语 Prompt (100-150字)
# ==========================================
def get_llm_recommendation(api_key, base_url, model, top1, top2, user_loc):
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        prompt = f"""
        你现在是同济大学一位热心且十分了解校园路线的‘老学长’。
        请根据算法结果，向在【{user_loc}】刚下课的学弟学妹精简对比推荐今天的前两名菜品。
        
        首选(Top 1): 【{top1['Dish_Name']}】(位于{top1['Canteen']}, 距离{top1['Distance_Meters']}米, 最终{top1['Score']}分)。特色: {top1['Feature_Tags']}
        备选(Top 2): 【{top2['Dish_Name']}】(位于{top2['Canteen']}, 距离{top2['Distance_Meters']}米, 最终{top2['Score']}分)。特色: {top2['Feature_Tags']}
        
        要求：
        1. 篇幅严格控制在 100 到 150 字之间。文字要精炼、口语化。
        2. 先一句话点出最完美的 Top 1 为什么适合当前位置；再用一句话说明如果不选第一，去吃 Top 2 有什么独特的性价比或口味优势。
        3. 不要写任何多余的开头语，直接以学长口吻输出推荐核心。
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位说话利落、直奔主题的同济大学干饭学长。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"老学长在连接大模型时被拥挤的食堂卡住了... 错误提示: {e}"

# ==========================================
# 5. 按钮触发与双表联动渲染
# ==========================================
if st.button("帮我决定吃什么！", type="primary"):
    try:
        raw_meals = pd.read_excel("meals.xlsx")
        raw_distance = pd.read_excel("distance.xlsx")
        raw_meals.columns = raw_meals.columns.str.strip()
        
        result_df = run_recommendation(raw_meals, raw_distance, current_location, budget, time_left, taste_preference)
        
        if result_df.empty:
            st.warning("⚠️ 报告老学长：在你的预算内，目前食堂没有能买到的菜品，建议调高预算！")
        elif len(result_df) < 2:
            st.warning("⚠️ 筛选出的备选菜品不足 2 种，请放宽口味偏好或调高预算！")
        else:
            top1 = result_df.iloc[0].to_dict()
            top2 = result_df.iloc[1].to_dict()
            
            st.success("🎯 算法最佳双首选已锁定！")
            
            # --- 头部高亮卡片展示 ---
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"### 🥇 首选：{top1['Dish_Name']}")
                st.caption(f"📍 {top1['Canteen']} · 💰 {top1['Price']}元 · 📈 {top1['Score']}分")
            with col_b:
                st.markdown(f"### 🥈 备选：{top2['Dish_Name']}")
                st.caption(f"📍 {top2['Canteen']} · 💰 {top2['Price']}元 · 📈 {top2['Score']}分")
            
            st.divider()
            
            # --- 大模型推荐语展示区 ---
            if api_key:
                with st.spinner("🎒 老学长正在帮你盘算干饭路线..."):
                    llm_speech = get_llm_recommendation(api_key, base_url, model_name, top1, top2, current_location)
                    st.chat_message("assistant", avatar="🎒").write(llm_speech)
            else:
                st.warning("💡 提示：侧边栏未配置 API Key。")
            
            # --- 详细打分数据表 ---
            with st.expander("📊 查看全校完整菜品精细化得分表 (连续多次点击按钮，会触发趣味探索分数更新哦！）"):
                st.dataframe(
                    result_df[["Dish_Name", "Canteen", "Price", "Distance_Meters", "Score", "Reason", "Feature_Tags"]],
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"❌ 系统错误: {e}")