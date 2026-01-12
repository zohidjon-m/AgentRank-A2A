#  AgentRank: Intelligent Agent Selection Layer for A2A Ecosystems
*A domain-aware, learning, exploration-enabled ranking engine for multi-agent systems.*

##  Overview
Modern multi-agent environments often contain **multiple agents that provide similar capabilities** (e.g., summarizers, translators, recruiters).  
The A2A protocol defines how agents communicate, but it does **not** define how to select the best agent.

**AgentRank** solves this by ranking agents using:
- performance metrics  
- domain policies  
- exploration techniques  
- learning from logs  

##  Features
- Intelligent agent selection  
- Config-driven scoring  
- Domain-aware policies  
- UCB exploration  
- Dynamic learning  
- A2A integration  

##  Architecture
```
Client Agent → AgentRank Service → Best Agent → A2A Request → Logs
```

## Metrics
- **Success Rate (SR)**  
- **Quality Score (QS)**  
- **Latency Score (LS)**  
- **Failure Rate (FR)**  

##  Ranking Algorithm
### 1. Base Score
```
base_score = wSR·SR + wQS·QS + wLS·LS + wFR·FR
```

### 2. Exploration Bonus (UCB)
```
exploration = α * sqrt( ln(1+N) / (1+n_a) )
```

### 3. Final Score
```
final_score = base_score + exploration
```

##  Project Structure
```
│ run_demo.py
│ agent_client.py
│ agent_rank_service.py
│ log_store.py
│ domain_registry.py
│ a2a_protocol.py
└ agents/
```

##  Running
```
python run_demo.py
```

##  Conclusion
AgentRank transforms a static multi-agent system into a **self-optimizing, intelligent, scalable ecosystem**.
