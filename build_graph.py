import json
import os
import re
import unicodedata
import pandas as pd


def safe_id(prefix: str, raw: str) -> str:
    """把任意字符串转成安全的 id（避免特殊字符导致节点冲突/连边错误）"""
    if pd.isna(raw):
        raw = "NA"
    raw = str(raw).strip()
    raw = unicodedata.normalize("NFKC", raw)
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^0-9a-zA-Z_\-\.]", "_", raw)
    if not raw:
        raw = "EMPTY"
    return f"{prefix}{raw}"


def build_graph_data(
    db_model_file: str,
    model_feature_file: str,
    out_json: str = "graph_data.json",
):
    db_model_df = pd.read_csv(db_model_file)
    model_feature_df = pd.read_csv(model_feature_file)

    # 必要列检查
    for col in ["database", "model", "usage_count"]:
        if col not in db_model_df.columns:
            raise ValueError(f"db_model_usage.csv 缺少列: {col}")

    # 这里要求 model_feature_freq.csv 必须有 database（你已采用方案二）
    for col in ["database", "model", "feature", "frequency"]:
        if col not in model_feature_df.columns:
            raise ValueError(f"model_feature_freq.csv 缺少列: {col}")

    # 数值清洗
    db_model_df["usage_count"] = pd.to_numeric(db_model_df["usage_count"], errors="coerce").fillna(0)
    model_feature_df["frequency"] = pd.to_numeric(model_feature_df["frequency"], errors="coerce").fillna(0)

    db_model_df = db_model_df.dropna(subset=["database", "model"])
    model_feature_df = model_feature_df.dropna(subset=["database", "model", "feature"])

    # 统一字符串
    db_model_df["database"] = db_model_df["database"].astype(str)
    db_model_df["model"] = db_model_df["model"].astype(str)

    model_feature_df["database"] = model_feature_df["database"].astype(str)
    model_feature_df["model"] = model_feature_df["model"].astype(str)
    model_feature_df["feature"] = model_feature_df["feature"].astype(str)

    # 口径校验：model_feature 中出现的 (database, model) 必须能在 db_model_usage 中找到
    dm_keys = set(zip(db_model_df["database"], db_model_df["model"]))
    mf_keys = set(zip(model_feature_df["database"], model_feature_df["model"]))
    missing = sorted(mf_keys - dm_keys)
    if missing:
        msg = "model_feature_freq.csv 中存在 (database, model) 组合，但 db_model_usage.csv 中找不到对应记录：\n"
        for db, m in missing[:50]:
            msg += f"- database={db}, model={m}\n"
        if len(missing) > 50:
            msg += f"... 共 {len(missing)} 条（仅展示前 50）\n"
        raise ValueError(msg)

    # ===== 全局节点集合（关键变化：model/feature 全局唯一）=====
    databases = sorted(db_model_df["database"].unique().tolist())
    models = sorted(pd.unique(pd.concat([db_model_df["model"], model_feature_df["model"]])).tolist())
    features = sorted(model_feature_df["feature"].unique().tolist())

    nodes = []
    node_index = set()

    def add_node(node):
        if node["id"] in node_index:
            return
        node_index.add(node["id"])
        nodes.append(node)

    # 数据库节点
    for db in databases:
        add_node({
            "id": safe_id("db_", db),
            "label": db,
            "group": "database",
            "title": f"数据库: {db}",
            "meta": {"type": "database", "name": db},
        })

    # 模型节点（全局唯一）
    for m in models:
        add_node({
            "id": safe_id("model_", m),
            "label": m,
            "group": "model",
            "title": f"模型: {m}",
            "meta": {"type": "model", "name": m},
        })

    # 特征节点（全局唯一）
    for f in features:
        add_node({
            "id": safe_id("feature_", f),
            "label": f,
            "group": "feature",
            "title": f"特征: {f}",
            "meta": {"type": "feature", "name": f},
        })

    # ===== edges =====
    edges = []

    # db -> model（每个库连到共享 model 节点）
    for _, r in db_model_df.iterrows():
        db = r["database"]
        m = r["model"]
        usage = float(r["usage_count"])
        edges.append({
            "id": f"dbmodel:{safe_id('db_', db)}->{safe_id('model_', m)}",
            "from": safe_id("db_", db),
            "to": safe_id("model_", m),
            "type": "db_model",
            "value": usage,
            "title": f"[{db}] 使用次数: {usage:g}",
            "label": str(int(usage)) if usage.is_integer() else f"{usage:g}",
            "meta": {"database": db, "model": m},
        })

    # model -> feature（同一 model-feature 可以因为不同 db 出现多条边；用 meta.database 保留口径）
    for idx, r in model_feature_df.iterrows():
        db = r["database"]
        m = r["model"]
        f = r["feature"]
        freq = float(r["frequency"])

        mid = safe_id("model_", m)
        fid = safe_id("feature_", f)

        edges.append({
            "id": f"mf:{mid}->{fid}:db={safe_id('', db)}:{idx}",
            "from": mid,
            "to": fid,
            "type": "model_feature",
            "value": freq,
            "title": f"[{db}] {m} 使用特征 {f} 次数: {freq:g}",
            "label": str(int(freq)) if freq.is_integer() else f"{freq:g}",
            "meta": {"database": db, "model": m, "feature": f},
        })

    payload = {
        "meta": {
            "databases": databases,
            "models": models,
            "features_count": len(features),
            "edges_count": len(edges),
            "note": "model/feature 为全局共享节点；model_feature 边用 meta.database 区分不同数据库口径",
        },
        "nodes": nodes,
        "edges": edges,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"✅ 已生成 {out_json}")
    print(f"- nodes: {len(nodes)}")
    print(f"- edges: {len(edges)}")
    print(f"- databases: {len(databases)}, models(global): {len(models)}, features(global): {len(features)}")


if __name__ == "__main__":
    db_model_csv = "D:/Download/Academic/pycharmworkspace/aaa/sjk/sjk_file/db_model_usage.csv"
    model_feature_csv = "D:/Download/Academic/pycharmworkspace/aaa/sjk/sjk_file/model_feature_freq.csv"

    if not os.path.exists(db_model_csv):
        raise FileNotFoundError(db_model_csv)
    if not os.path.exists(model_feature_csv):
        raise FileNotFoundError(model_feature_csv)

    build_graph_data(db_model_csv, model_feature_csv, out_json="graph_data.json")
