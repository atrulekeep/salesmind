"""作者脚本：构造 100 条评测集并写入 eval/cases.jsonl。

设计要点（对齐 spec §8.1）：
- 6 类分布：fact30 / entity20 / paraphrase20 / multihop15 / temporal10 / unanswerable5
- relevant_chunk_ids 精确到真实 chunk_id（来自 eval/chunk_map.json）
- temporal 题只把“现行版”chunk 作 gold；已废止 chunk 是陷阱，不应进 gold
- unanswerable 题 gold 为空、should_refuse=true
- answer_keywords 为“必需关键词”，metrics 要求全部出现才算命中
- corpus v1 ↔ eval v1 版本锁定

跑法：.venv/bin/python -m eval._author_cases
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CHUNK_MAP = EVAL_DIR / "chunk_map.json"
CASES = EVAL_DIR / "cases.jsonl"

# ──────────────────────────────────────────────
# 100 条标注
# ──────────────────────────────────────────────
CASE_LIST: list[dict] = [
    # ── fact（30）：单点事实，1 个 gold ───────────────────
    {"id": "eval-001", "category": "fact", "question": "专业版的年度订阅价是多少元？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["19800"], "should_refuse": False},
    {"id": "eval-002", "category": "fact", "question": "旗舰版年费是多少？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["39800"], "should_refuse": False},
    {"id": "eval-003", "category": "fact", "question": "标准版包年价多少？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["9800"], "should_refuse": False},
    {"id": "eval-004", "category": "fact", "question": "合同金额满 10 万享受几折？",
     "relevant_chunk_ids": ["pricing_2026#2"], "answer_keywords": ["8折"], "should_refuse": False},
    {"id": "eval-005", "category": "fact", "question": "合同金额在 2 万到 5 万之间折扣几折？",
     "relevant_chunk_ids": ["pricing_2026#2"], "answer_keywords": ["9折"], "should_refuse": False},
    {"id": "eval-006", "category": "fact", "question": "老客户续费享几折？",
     "relevant_chunk_ids": ["pricing_2026#4"], "answer_keywords": ["95折"], "should_refuse": False},
    {"id": "eval-007", "category": "fact", "question": "续费需在到期前多少天内启动？",
     "relevant_chunk_ids": ["pricing_2026#4"], "answer_keywords": ["60"], "should_refuse": False},
    {"id": "eval-008", "category": "fact", "question": "标准报价含税吗？增值税几个点？",
     "relevant_chunk_ids": ["sales_faq#2"], "answer_keywords": ["6%"], "should_refuse": False},
    {"id": "eval-009", "category": "fact", "question": "专业版包年包含多少个账号？",
     "relevant_chunk_ids": ["sales_faq#1"], "answer_keywords": ["30"], "should_refuse": False},
    {"id": "eval-010", "category": "fact", "question": "专业版超出账号按多少元每个每年增购？",
     "relevant_chunk_ids": ["sales_faq#1"], "answer_keywords": ["600"], "should_refuse": False},
    {"id": "eval-011", "category": "fact", "question": "SaaS 标准版签约后几个工作日内开通？",
     "relevant_chunk_ids": ["sales_faq#3"], "answer_keywords": ["3"], "should_refuse": False},
    {"id": "eval-012", "category": "fact", "question": "专业版标准实施周期是几个工作日？",
     "relevant_chunk_ids": ["implementation_guide#1"], "answer_keywords": ["10"], "should_refuse": False},
    {"id": "eval-013", "category": "fact", "question": "旗舰版私有化部署实施周期几个工作日？",
     "relevant_chunk_ids": ["implementation_guide#1"], "answer_keywords": ["30"], "should_refuse": False},
    {"id": "eval-014", "category": "fact", "question": "开通后几个工作日内可申请无理由退款？",
     "relevant_chunk_ids": ["sales_faq#5"], "answer_keywords": ["7"], "should_refuse": False},
    {"id": "eval-015", "category": "fact", "question": "低于几折的特价申请须走特价审批流程？",
     "relevant_chunk_ids": ["pricing_2026#3"], "answer_keywords": ["8折"], "should_refuse": False},
    {"id": "eval-016", "category": "fact", "question": "合同金额满 10 万的 8 折须谁审批？",
     "relevant_chunk_ids": ["pricing_2026#2"], "answer_keywords": ["销售总监"], "should_refuse": False},
    {"id": "eval-017", "category": "fact", "question": "标准合同付款首付款比例是多少？",
     "relevant_chunk_ids": ["contract_sla#0"], "answer_keywords": ["50%"], "should_refuse": False},
    {"id": "eval-018", "category": "fact", "question": "系统上线验收后付款上线款比例是多少？",
     "relevant_chunk_ids": ["contract_sla#0"], "answer_keywords": ["30%"], "should_refuse": False},
    {"id": "eval-019", "category": "fact", "question": "合同尾款比例是多少？上线后多少工作日内付？",
     "relevant_chunk_ids": ["contract_sla#0"], "answer_keywords": ["20%", "90"], "should_refuse": False},
    {"id": "eval-020", "category": "fact", "question": "SaaS 版月度可用性承诺不低于多少？",
     "relevant_chunk_ids": ["contract_sla#1"], "answer_keywords": ["99.9%"], "should_refuse": False},
    {"id": "eval-021", "category": "fact", "question": "P1 级故障的响应时效是多久？",
     "relevant_chunk_ids": ["contract_sla#2"], "answer_keywords": ["30"], "should_refuse": False},
    {"id": "eval-022", "category": "fact", "question": "P1 级故障的恢复时效是多久？",
     "relevant_chunk_ids": ["contract_sla#2"], "answer_keywords": ["4"], "should_refuse": False},
    {"id": "eval-023", "category": "fact", "question": "P2 级故障响应时效是多久？",
     "relevant_chunk_ids": ["contract_sla#2"], "answer_keywords": ["2"], "should_refuse": False},
    {"id": "eval-024", "category": "fact", "question": "SaaS 版密钥多少天轮换一次？",
     "relevant_chunk_ids": ["security_compliance#0"], "answer_keywords": ["90"], "should_refuse": False},
    {"id": "eval-025", "category": "fact", "question": "SaaS 版数据传输采用什么加密？",
     "relevant_chunk_ids": ["security_compliance#0"], "answer_keywords": ["TLS 1.3"], "should_refuse": False},
    {"id": "eval-026", "category": "fact", "question": "SaaS 版数据落盘采用什么加密算法？",
     "relevant_chunk_ids": ["security_compliance#0"], "answer_keywords": ["AES-256"], "should_refuse": False},
    {"id": "eval-027", "category": "fact", "question": "银牌代理商返点比例是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["8%"], "should_refuse": False},
    {"id": "eval-028", "category": "fact", "question": "金牌代理商返点比例是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["12%"], "should_refuse": False},
    {"id": "eval-029", "category": "fact", "question": "白金代理商返点比例是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["15%"], "should_refuse": False},
    {"id": "eval-030", "category": "fact", "question": "代理商项目报备成功后享多少天保护期？",
     "relevant_chunk_ids": ["partner_channel#2"], "answer_keywords": ["30"], "should_refuse": False},

    # ── entity（20）：精确实体（SKU/名称/数字）───────────
    {"id": "eval-031", "category": "entity", "question": "专业版的 SKU 编号是什么？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["ZC-PRO-2026"], "should_refuse": False},
    {"id": "eval-032", "category": "entity", "question": "旗舰版的 SKU 编号是什么？",
     "relevant_chunk_ids": ["product_catalog#2"], "answer_keywords": ["ZC-FLG-2026"], "should_refuse": False},
    {"id": "eval-033", "category": "entity", "question": "标准版的 SKU 编号是什么？",
     "relevant_chunk_ids": ["product_catalog#2"], "answer_keywords": ["ZC-STD-2026"], "should_refuse": False},
    {"id": "eval-034", "category": "entity", "question": "50 人以下销售团队适用哪个套餐及 SKU？",
     "relevant_chunk_ids": ["product_catalog#2"], "answer_keywords": ["标准版", "ZC-STD-2026"], "should_refuse": False},
    {"id": "eval-035", "category": "entity", "question": "300 人以上集团客户适用哪个套餐？",
     "relevant_chunk_ids": ["product_catalog#2"], "answer_keywords": ["旗舰版", "ZC-FLG-2026"], "should_refuse": False},
    {"id": "eval-036", "category": "entity", "question": "智策云CRM 包含哪四大模块？",
     "relevant_chunk_ids": ["product_catalog#0"], "answer_keywords": ["销售自动化", "商机管道"], "should_refuse": False},
    {"id": "eval-037", "category": "entity", "question": "合同与回款模块包含哪些线上化能力？",
     "relevant_chunk_ids": ["product_catalog#5"], "answer_keywords": ["回款计划"], "should_refuse": False},
    {"id": "eval-038", "category": "entity", "question": "城商行案例选用了哪个套餐？",
     "relevant_chunk_ids": ["case_studies#2"], "answer_keywords": ["旗舰版"], "should_refuse": False},
    {"id": "eval-039", "category": "entity", "question": "汽车零部件案例的销售团队有多少人？",
     "relevant_chunk_ids": ["case_studies#1"], "answer_keywords": ["120"], "should_refuse": False},
    {"id": "eval-040", "category": "entity", "question": "汽车零部件案例回款周期从多少天缩短到多少天？",
     "relevant_chunk_ids": ["case_studies#1"], "answer_keywords": ["68", "42"], "should_refuse": False},
    {"id": "eval-041", "category": "entity", "question": "城商行案例客户经理人均管户数从多少提升到多少？",
     "relevant_chunk_ids": ["case_studies#2"], "answer_keywords": ["80", "150"], "should_refuse": False},
    {"id": "eval-042", "category": "entity", "question": "HR SaaS 案例的年费是多少元？",
     "relevant_chunk_ids": ["case_studies#3"], "answer_keywords": ["9800"], "should_refuse": False},
    {"id": "eval-043", "category": "entity", "question": "智策云CRM 已取得哪三项合规认证？",
     "relevant_chunk_ids": ["security_compliance#1"], "answer_keywords": ["等保", "SOC", "ISO"], "should_refuse": False},
    {"id": "eval-044", "category": "entity", "question": "内部员工访问生产数据的审计日志留存多少天？",
     "relevant_chunk_ids": ["security_compliance#3"], "answer_keywords": ["180"], "should_refuse": False},
    {"id": "eval-045", "category": "entity", "question": "纷享销客的标准起步价是多少元？",
     "relevant_chunk_ids": ["competitor_comparison#1"], "answer_keywords": ["16800"], "should_refuse": False},
    {"id": "eval-046", "category": "entity", "question": "Salescloud Enterprise 起步价约多少美元？",
     "relevant_chunk_ids": ["competitor_comparison#1"], "answer_keywords": ["12000"], "should_refuse": False},
    {"id": "eval-047", "category": "entity", "question": "销售易专业版标准起步价是多少元？",
     "relevant_chunk_ids": ["competitor_comparison#1"], "answer_keywords": ["19800"], "should_refuse": False},
    {"id": "eval-048", "category": "entity", "question": "金牌代理商年到账销售额门槛是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["60"], "should_refuse": False},
    {"id": "eval-049", "category": "entity", "question": "白金代理商年到账销售额门槛是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["120"], "should_refuse": False},
    {"id": "eval-050", "category": "entity", "question": "技术架构师在哪个套餐标配配备？",
     "relevant_chunk_ids": ["implementation_guide#2"], "answer_keywords": ["旗舰版"], "should_refuse": False},

    # ── paraphrase（20）：同义改写，复用对应 fact 的 gold ──
    {"id": "eval-051", "category": "paraphrase", "question": "专业版包年价是多少？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["19800"], "should_refuse": False},
    {"id": "eval-052", "category": "paraphrase", "question": "旗舰版年度订阅价多少？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["39800"], "should_refuse": False},
    {"id": "eval-053", "category": "paraphrase", "question": "续约有没有优惠折扣？",
     "relevant_chunk_ids": ["pricing_2026#4"], "answer_keywords": ["95折"], "should_refuse": False},
    {"id": "eval-054", "category": "paraphrase", "question": "老客户到期前续约给几折？",
     "relevant_chunk_ids": ["pricing_2026#4"], "answer_keywords": ["95折"], "should_refuse": False},
    {"id": "eval-055", "category": "paraphrase", "question": "多加账号每个多少钱一年？",
     "relevant_chunk_ids": ["sales_faq#1"], "answer_keywords": ["600"], "should_refuse": False},
    {"id": "eval-056", "category": "paraphrase", "question": "超员增购账号单价多少？",
     "relevant_chunk_ids": ["sales_faq#1"], "answer_keywords": ["600"], "should_refuse": False},
    {"id": "eval-057", "category": "paraphrase", "question": "开票税率是多少？",
     "relevant_chunk_ids": ["sales_faq#2"], "answer_keywords": ["6%"], "should_refuse": False},
    {"id": "eval-058", "category": "paraphrase", "question": "增值税专用发票几个点？",
     "relevant_chunk_ids": ["sales_faq#2"], "answer_keywords": ["6%"], "should_refuse": False},
    {"id": "eval-059", "category": "paraphrase", "question": "标准版签约后多久能开通？",
     "relevant_chunk_ids": ["sales_faq#3"], "answer_keywords": ["3"], "should_refuse": False},
    {"id": "eval-060", "category": "paraphrase", "question": "标准版的开通时效是几天？",
     "relevant_chunk_ids": ["sales_faq#3"], "answer_keywords": ["3"], "should_refuse": False},
    {"id": "eval-061", "category": "paraphrase", "question": "不满意能退款吗？",
     "relevant_chunk_ids": ["sales_faq#5"], "answer_keywords": ["7"], "should_refuse": False},
    {"id": "eval-062", "category": "paraphrase", "question": "几天内可以无理由退款？",
     "relevant_chunk_ids": ["sales_faq#5"], "answer_keywords": ["7"], "should_refuse": False},
    {"id": "eval-063", "category": "paraphrase", "question": "合同满 10 万给几折？",
     "relevant_chunk_ids": ["pricing_2026#2"], "answer_keywords": ["8折"], "should_refuse": False},
    {"id": "eval-064", "category": "paraphrase", "question": "大单折扣几折封顶？",
     "relevant_chunk_ids": ["pricing_2026#2"], "answer_keywords": ["8折"], "should_refuse": False},
    {"id": "eval-065", "category": "paraphrase", "question": "能不能按月付费？",
     "relevant_chunk_ids": ["sales_faq#0"], "answer_keywords": ["包年"], "should_refuse": False},
    {"id": "eval-066", "category": "paraphrase", "question": "支持月度付费吗？",
     "relevant_chunk_ids": ["sales_faq#0"], "answer_keywords": ["包年"], "should_refuse": False},
    {"id": "eval-067", "category": "paraphrase", "question": "首付款要付几成？",
     "relevant_chunk_ids": ["contract_sla#0"], "answer_keywords": ["50%"], "should_refuse": False},
    {"id": "eval-068", "category": "paraphrase", "question": "签约后先付多少比例？",
     "relevant_chunk_ids": ["contract_sla#0"], "answer_keywords": ["50%"], "should_refuse": False},
    {"id": "eval-069", "category": "paraphrase", "question": "系统承诺的可用率是多少？",
     "relevant_chunk_ids": ["contract_sla#1"], "answer_keywords": ["99.9%"], "should_refuse": False},
    {"id": "eval-070", "category": "paraphrase", "question": "服务可用性承诺多少？",
     "relevant_chunk_ids": ["contract_sla#1"], "answer_keywords": ["99.9%"], "should_refuse": False},

    # ── multihop（15）：跨文档多跳，2-3 个 gold ─────────
    {"id": "eval-071", "category": "multihop", "question": "专业版包年含 30 个账号，超出按 600 元每个每年增购，那么专业版 50 个账号一年共多少钱？",
     "relevant_chunk_ids": ["pricing_2026#1", "sales_faq#1"], "answer_keywords": ["19800", "600", "31800"], "should_refuse": False},
    {"id": "eval-072", "category": "multihop", "question": "旗舰版原价 39800 元，合同金额满 10 万才享 8 折，单买一台旗舰版能享受 8 折吗？",
     "relevant_chunk_ids": ["pricing_2026#1", "pricing_2026#2"], "answer_keywords": ["39800", "10万", "8折"], "should_refuse": False},
    {"id": "eval-073", "category": "multihop", "question": "城商行因数据出境合规选了私有化部署，它用的是哪个套餐、实施周期几个工作日？",
     "relevant_chunk_ids": ["case_studies#2", "implementation_guide#1"], "answer_keywords": ["旗舰版", "30"], "should_refuse": False},
    {"id": "eval-074", "category": "multihop", "question": "汽车零部件案例上了专业版，按续费 95 折，第二年续费应付多少元？",
     "relevant_chunk_ids": ["case_studies#1", "pricing_2026#4", "pricing_2026#1"], "answer_keywords": ["19800", "95", "18810"], "should_refuse": False},
    {"id": "eval-075", "category": "multihop", "question": "标准版年费 9800 元、3 工作日开通，按付款首付 50% 计，首付款是多少元？",
     "relevant_chunk_ids": ["pricing_2026#1", "sales_faq#3", "contract_sla#0"], "answer_keywords": ["9800", "50%", "4900"], "should_refuse": False},
    {"id": "eval-076", "category": "multihop", "question": "金牌代理商返点 12%，年代理销售额 60 万元，返点金额是多少元？",
     "relevant_chunk_ids": ["partner_channel#1", "partner_channel#3"], "answer_keywords": ["12%", "60", "72000"], "should_refuse": False},
    {"id": "eval-077", "category": "multihop", "question": "标准版售后仅享 P2/P3，旗舰版含 P1 全天候响应，标准版支持 P1 故障响应吗？",
     "relevant_chunk_ids": ["contract_sla#2", "contract_sla#3"], "answer_keywords": ["P2", "P3", "标准版"], "should_refuse": False},
    {"id": "eval-078", "category": "multihop", "question": "技术架构师仅旗舰版标配，城商行案例选了旗舰版，会配备技术架构师吗？",
     "relevant_chunk_ids": ["implementation_guide#2", "case_studies#2"], "answer_keywords": ["技术架构师", "旗舰版"], "should_refuse": False},
    {"id": "eval-079", "category": "multihop", "question": "专业版标准实施 10 工作日、年费 19800 元，按首付 50% 计，首付多少元且实施多久？",
     "relevant_chunk_ids": ["pricing_2026#1", "contract_sla#0", "implementation_guide#1"], "answer_keywords": ["19800", "9900", "10"], "should_refuse": False},
    {"id": "eval-080", "category": "multihop", "question": "旗舰版私有化数据留客户内网且不出境，旗舰版年费 39800 元，续费 95 折应付多少元？",
     "relevant_chunk_ids": ["security_compliance#2", "pricing_2026#1", "pricing_2026#4"], "answer_keywords": ["39800", "95", "37810"], "should_refuse": False},
    {"id": "eval-081", "category": "multihop", "question": "Salesforce 起步约 12000 美元且数据出境，智策云标准版 9800 元，二者适用差异在哪？",
     "relevant_chunk_ids": ["competitor_comparison#1", "competitor_comparison#2"], "answer_keywords": ["12000", "9800", "数据出境"], "should_refuse": False},
    {"id": "eval-082", "category": "multihop", "question": "银牌代理商返点 8%、报备保护期 30 天，某银牌代理成交 20 万元，返点金额是多少元？",
     "relevant_chunk_ids": ["partner_channel#2", "partner_channel#1", "partner_channel#3"], "answer_keywords": ["8%", "20", "16000"], "should_refuse": False},
    {"id": "eval-083", "category": "multihop", "question": "合规认证含 SOC 2 与 ISO 27001，漏洞走 P1，高危漏洞须几个工作日内修复？",
     "relevant_chunk_ids": ["security_compliance#1", "security_compliance#4"], "answer_keywords": ["7"], "should_refuse": False},
    {"id": "eval-084", "category": "multihop", "question": "合同与回款模块签约后自动生成回款节点，尾款 20% 须上线后多少工作日内付？",
     "relevant_chunk_ids": ["product_catalog#5", "contract_sla#0"], "answer_keywords": ["20%", "90"], "should_refuse": False},
    {"id": "eval-085", "category": "multihop", "question": "HR SaaS 案例用标准版、3 工作日开通、年费 9800 元，按首付 50% 计首付款多少元？",
     "relevant_chunk_ids": ["case_studies#3", "contract_sla#0"], "answer_keywords": ["9800", "50%", "4900"], "should_refuse": False},

    # ── temporal（10）：gold 只取现行版，已废止为陷阱 ────
    {"id": "eval-086", "category": "temporal", "question": "2026 现行版标准版年费是多少？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["9800"], "should_refuse": False},
    {"id": "eval-087", "category": "temporal", "question": "2026 现行版专业版年费是多少？",
     "relevant_chunk_ids": ["pricing_2026#1"], "answer_keywords": ["19800"], "should_refuse": False},
    {"id": "eval-088", "category": "temporal", "question": "现行合同满 10 万享几折？",
     "relevant_chunk_ids": ["pricing_2026#2"], "answer_keywords": ["8折"], "should_refuse": False},
    {"id": "eval-089", "category": "temporal", "question": "现行银牌代理商返点比例是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["8%"], "should_refuse": False},
    {"id": "eval-090", "category": "temporal", "question": "现行金牌代理商返点比例是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["12%"], "should_refuse": False},
    {"id": "eval-091", "category": "temporal", "question": "现行项目报备保护期是多少天？",
     "relevant_chunk_ids": ["partner_channel#2"], "answer_keywords": ["30"], "should_refuse": False},
    {"id": "eval-092", "category": "temporal", "question": "现行代理商对外报价底线几折？",
     "relevant_chunk_ids": ["partner_channel#4"], "answer_keywords": ["8折"], "should_refuse": False},
    {"id": "eval-093", "category": "temporal", "question": "现行白金代理商年到账门槛是多少？",
     "relevant_chunk_ids": ["partner_channel#1"], "answer_keywords": ["120"], "should_refuse": False},
    {"id": "eval-094", "category": "temporal", "question": "现行满 10 万折扣须谁审批？",
     "relevant_chunk_ids": ["pricing_2026#2"], "answer_keywords": ["销售总监"], "should_refuse": False},
    {"id": "eval-095", "category": "temporal", "question": "现行续费享几折？",
     "relevant_chunk_ids": ["pricing_2026#4"], "answer_keywords": ["95折"], "should_refuse": False},

    # ── unanswerable（5）：知识库不覆盖，应拒答 ──────────
    {"id": "eval-096", "category": "unanswerable", "question": "智策云员工的薪资是多少？",
     "relevant_chunk_ids": [], "answer_keywords": [], "should_refuse": True},
    {"id": "eval-097", "category": "unanswerable", "question": "公司最近一轮融资的估值是多少？",
     "relevant_chunk_ids": [], "answer_keywords": [], "should_refuse": True},
    {"id": "eval-098", "category": "unanswerable", "question": "销售总监的姓名和联系方式是什么？",
     "relevant_chunk_ids": [], "answer_keywords": [], "should_refuse": True},
    {"id": "eval-099", "category": "unanswerable", "question": "员工股票期权的行权价是多少？",
     "relevant_chunk_ids": [], "answer_keywords": [], "should_refuse": True},
    {"id": "eval-100", "category": "unanswerable", "question": "汽车零部件案例客户公司的真实名称是什么？",
     "relevant_chunk_ids": [], "answer_keywords": [], "should_refuse": True},
]


def _validate(cases: list[dict], known_ids: set[str]) -> None:
    assert len(cases) == 100, f"期望 100 条，实际 {len(cases)}"
    # 类别分布
    dist = Counter(c["category"] for c in cases)
    expected = {"fact": 30, "entity": 20, "paraphrase": 20,
                "multihop": 15, "temporal": 10, "unanswerable": 5}
    assert dist == expected, f"类别分布不符：{dist} vs {expected}"
    # id 唯一
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 100, "id 存在重复"
    # gold chunk_id 存在性
    for c in cases:
        for cid in c["relevant_chunk_ids"]:
            assert cid in known_ids, f"{c['id']} gold 不存在：{cid}"
    # unanswerable 必须空 gold + refuse
    for c in cases:
        if c["category"] == "unanswerable":
            assert c["relevant_chunk_ids"] == [], f"{c['id']} unanswerable 须空 gold"
            assert c["should_refuse"] is True, f"{c['id']} 须 refuse"
        else:
            assert len(c["relevant_chunk_ids"]) >= 1, f"{c['id']} 非 unanswerable 须有 gold"
    # 必需字段
    for c in cases:
        assert set(c) >= {"id", "category", "question",
                          "relevant_chunk_ids", "answer_keywords", "should_refuse"}
        assert isinstance(c["answer_keywords"], list)


def main() -> None:
    known_ids: set[str] = set()
    if CHUNK_MAP.exists():
        known_ids = set(json.loads(CHUNK_MAP.read_text("utf-8")).keys())
    _validate(CASE_LIST, known_ids)
    with CASES.open("w", encoding="utf-8") as f:
        for c in CASE_LIST:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✓ 已写入 {CASES}（{len(CASE_LIST)} 条）")
    print("  类别分布：", dict(Counter(c["category"] for c in CASE_LIST)))
    print(f"  已知 chunk_id 数：{len(known_ids)}")


if __name__ == "__main__":
    main()
