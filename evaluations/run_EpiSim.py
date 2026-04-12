import numpy as np
np.random.seed(0)

class SparseGraph:
    """【病毒传播人际网络拓扑图】：一个简单的无向图数据结构，用来表示谁跟谁挨过边、串过门了"""
    def __init__(self):
        self.graph = {}

    def add_node(self, node_id, attributes=None):
        if node_id not in self.graph:
            self.graph[node_id] = set()

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.graph[node1].add(node2)
        
    def get_all_node_ids(self):
        return list(self.graph.keys())

    def get_neighbors(self, node_id):
        return self.graph.get(node_id, set())

def calculate_seq_per_poi(dataset, t_day, t_granularity):
    """统计每家店每天几点几分接待了哪些客人（也就是潜在的病毒集散地打卡点）"""
    checkin_seq_per_poi = {}
    for uid, seq in enumerate(dataset):
        checkin_hour = (np.array(seq['arrival_times']) * t_granularity).astype(int)
        checkin_time = seq['arrival_times']
        for pid, poi in enumerate(seq['checkins']):
            if poi not in checkin_seq_per_poi.keys():
                checkin_seq_per_poi[poi] = {str(i+1): [] for i in range(t_day)}
            checkin_seq_per_poi[poi][str((checkin_hour[pid] // 24) + 1)].append((uid, checkin_time[pid]))
    return checkin_seq_per_poi

def construct_network_for_epidemic_simulation(poi_checkin_data, t_day):
    """
    【密接排查网】：把人与人之间串在一起。
    逻辑是只要你在同样的时间出现在了同样的店铺，我就牵一根线，当你俩是互相交叉密切接触者。
    """
    graphs = [SparseGraph() for i in range(t_day)]
    for poi in poi_checkin_data.keys():
        for i in range(t_day):
            seq_ids_time_list = poi_checkin_data[poi][str(i+1)]
            sorted_seq_ids_time_list = sorted(seq_ids_time_list, key=lambda x: x[1])
            if len(sorted_seq_ids_time_list) > 1:
                for j in range(len(sorted_seq_ids_time_list)):
                    for k in range(len(sorted_seq_ids_time_list)):
                        # 只要是两个人不同，我就在今天建他俩的连通边
                        if sorted_seq_ids_time_list[j][0] != sorted_seq_ids_time_list[k][0]:
                            graphs[i].add_edge(sorted_seq_ids_time_list[j][0], sorted_seq_ids_time_list[k][0])
                            graphs[i].add_edge(sorted_seq_ids_time_list[k][0], sorted_seq_ids_time_list[j][0])
    return graphs

class Global_epidemic_info:
    """【传染病四大天王隔离站】：传统的 SIR 经典传染病控制站舱室统计模块。"""
    def __init__(self, is_covid19):
        # 传染病的各特征率：密切接触率c，传播期T，潜伏期T_i等
        if is_covid19:
            self.c = 0.2
            self.T = 5.8
            self.T_i = 5.2
            self.T_f = 11
            self.R_0 = 2.2 # 基本传染数
            self.beta = self.R_0 / self.T  # 感染速率
            self.alpha = 1 / self.T_i      # 发病速率
            self.r = 1 / self.T_f          # 康复率
        else: # 普通流感
            self.c = 0.2
            self.beta = 0.402
            self.alpha = 0.526
            self.r = 0.244

        self.susceptible = 0  # S：易感者(正常人)
        self.exposed = 50     # E：潜伏期的人 (0号病人初始投放 50 个)
        self.infected = 0     # I：确诊患病者
        self.recovered = 0    # R：痊愈拥有抗体者

        self.susceptible_list = [self.susceptible]
        self.exposed_list = [self.exposed]
        self.infected_list = [self.infected]
        self.recovered_list = [self.recovered]

        self.susceptible_user = []
        self.exposed_user = []
        self.infected_user = []
        self.recovered_user = []

    def update(self):
        # 每天收摊时点名核算各区多少人
        self.susceptible_list.append(len(self.susceptible_user))
        self.exposed_list.append(len(self.exposed_user))
        self.infected_list.append(len(self.infected_user))
        self.recovered_list.append(len(self.recovered_user))

def epidemic_simulator(G, init_exposed_num, cycles, is_covid19):
    """【死神之手沙盘推演游戏】：模拟投放了 0号病人后，随日子更迭城市全员的生化交叉变化趋势"""
    ramdom_matrix = np.random.rand(10000000)
    ramdom_index = 0
    epidemic_record = Global_epidemic_info(is_covid19=is_covid19)
    
    # 盲抓一把人当倒霉蛋（零号病人）
    day1_user_list = list(G[0].get_all_node_ids())
    exposed_user_init = np.random.choice(day1_user_list, size=init_exposed_num, replace=False).tolist()
    epidemic_record.exposed_user.extend(exposed_user_init)
    epidemic_record.exposed_user = list(set(epidemic_record.exposed_user))
    epidemic_record.susceptible_user = []

    for cycle in range(cycles):
        for i in range(len(G)):
            susceptible_new = []
            # 凡是跟潜伏者和发病者有过邻居网络关系的，全都拉低至易感白名单边缘
            for exposed_u in epidemic_record.exposed_user:
                susceptible_new.extend(list(G[i].get_neighbors(exposed_u)))   
            for infected_u in epidemic_record.infected_user:
                susceptible_new.extend(list(G[i].get_neighbors(infected_u)))         
            susceptible_new = list(set(susceptible_new))
            susceptible_used = []
            
            for u in susceptible_new: # 去除掉已经中招或者有抗体的高贵血统
                if (u in epidemic_record.infected_user) or (u in epidemic_record.exposed_user) or (u in epidemic_record.recovered_user):
                    pass
                else:
                    susceptible_used.append(u)
            epidemic_record.susceptible_user.extend(susceptible_used)
            epidemic_record.susceptible_user = list(set(epidemic_record.susceptible_user))
            
            # 第一阶段：易感 -> 变潜伏（概率判定是否被成功飞沫传染）
            for susceptible_u in epidemic_record.susceptible_user:
                if ramdom_matrix[ramdom_index] <= (epidemic_record.c * epidemic_record.beta):
                    epidemic_record.susceptible_user.remove(susceptible_u)
                    epidemic_record.exposed_user.append(susceptible_u)
                ramdom_index += 1
                
            # 第二阶段：潜伏 -> 变发病确诊（潜伏期经过的时间发作率突破防线）
            for exposed_u in epidemic_record.exposed_user:
                if ramdom_matrix[ramdom_index] <= epidemic_record.alpha:
                    epidemic_record.exposed_user.remove(exposed_u)
                    epidemic_record.infected_user.append(exposed_u)
                ramdom_index += 1
                
            # 第三阶段：确诊 -> 变康复出院拥有抗体（抵抗力战胜了病毒）
            for infected_u in epidemic_record.infected_user:
                if ramdom_matrix[ramdom_index] <= epidemic_record.r:
                    epidemic_record.recovered_user.append(infected_u)
                    epidemic_record.infected_user.remove(infected_u)
                ramdom_index += 1
                
            # 今天天黑了，提交今日各路大军剩余人头报表
            epidemic_record.update()
            epidemic_record.susceptible_user = []
            
    return epidemic_record.exposed_list, epidemic_record.infected_list, epidemic_record.recovered_list

def run_simulator(datalist, init_exposed_num, exp_num=15, cycles=12, is_covid19=True, t_day=1, t_granularity=1):
    all_data_result_exposed, all_data_result_infected, all_data_result_recovered = [], [], []

    for idx, data in enumerate(datalist):
        checkin_seq_per_poi = calculate_seq_per_poi(data, t_day, t_granularity)
        # 用数据建出巨大的人人交互连串无理图网络
        G = construct_network_for_epidemic_simulation(checkin_seq_per_poi, t_day)
        result_exposed, result_infected, result_recovered = [], [], []
        
        # 为了防随机数暴走，一般实验跑 15 遍然后取折中平均曲线
        for i in range(exp_num):
            exposed, infected, recovered = epidemic_simulator(G, init_exposed_num, cycles, is_covid19)
            result_exposed.append(exposed[1:])
            result_infected.append(infected[1:])
            result_recovered.append(recovered[1:])
            
        result_exposed = np.mean(np.array(result_exposed), axis=0)
        result_infected = np.mean(np.array(result_infected), axis=0)
        result_recovered = np.mean(np.array(result_recovered), axis=0)

        all_data_result_exposed.append(result_exposed)
        all_data_result_infected.append(result_infected)
        all_data_result_recovered.append(result_recovered)

    real_exposed, generated_exposed = all_data_result_exposed[0], all_data_result_exposed[1]
    real_infected, generated_infected = all_data_result_infected[0], all_data_result_infected[1]
    real_recovered, generated_recovered = all_data_result_recovered[0], all_data_result_recovered[1]

    # 给人类足迹版沙盘，和造假版沙盘的两条感染曲线计算面积绝对差
    relative_difference_e = np.abs((real_exposed - generated_exposed) / (real_exposed))
    relative_difference_i = np.abs((real_infected - generated_infected) / (real_infected))
    relative_difference_r = np.abs((real_recovered - generated_recovered) / (real_recovered))

    return relative_difference_e, relative_difference_i, relative_difference_r

def run_EpiSim_task(test_data, generated_data, init_exposed_num=50, exp_num=15, cycles=12):
    """
    【疫情双轨沙盘实战检测评估器主路口】
    """
    datalist = [test_data, generated_data]
    # 先做新冠病毒毒性沙盘连考 (高烈度)
    re_e_c19, re_i_c19, re_r_c19 = run_simulator(datalist, init_exposed_num, exp_num, cycles, True)
    # 再玩把普通流感毒性沙盘连考 (低毒性)
    re_e_inf, re_i_inf, re_r_inf = run_simulator(datalist, init_exposed_num, exp_num, cycles, False)
    
    # 全部加一起取个均方差打总分板返回：
    results_relative = np.array([re_e_c19, re_i_c19, re_r_c19, re_e_inf, re_i_inf, re_r_inf])
    MAPE = np.mean(results_relative)
    MSPE = np.mean(np.square(results_relative))
    return MAPE, MSPE

"""
========================================================================
【给外行新手的通俗讲解】 -- run_EpiSim.py (流行病危机沙盘模拟器)
========================================================================
这简直是个绝妙且脑洞大开的模型质检方法。
想象一下，如果我们用 AI 造出了一堆只会天天窝在家里打游戏不出门的数据人。这叫不贴合实际。
为了测出这些造出来的数据到底具不具备典型的人类**社会大聚集及流动性**，
这里的开发者写了一个“病毒沙盘扩散游戏（类似于著名的瘟疫公司游戏）”。

他拿着人类自己的数据和 AI造出来的数据 分开建了两座城：
1. 建构起连接着几万个店面的互通交流网 (`construct_network`)
2. 向两座城市同时投放 50 个带有新冠剧毒和流感的中招零号病人 (`init_exposed_num`)
3. 随后每天靠着电脑随机扔骰子结合传染病医学论文里的潜伏发作参数(`is_covid19`)模拟两座城各自交叉感染了多少人，最终呈现多少张发病曲线。

如果 AI 数据造假水平过硬造得像模像样，那么这两座虚拟城市的**发病爆表高潮期和传播速率应该严丝合缝如出一辙**。相反，如果AI造的数据人都瞎乱窜，传播速度会发生极恐怖的偏离，这也就是最终通过 `MAPE` 报错反馈给总部的证据！
========================================================================
"""