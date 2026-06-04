# Changelog

Format: `<date> · <task-id> · <description>`

## 2026-06-04
- P1-05 · MasteryEngine 熟练度引擎（状态机自动升级、按方法连接、lookup_batch 单 SQL，34 单测 99% 覆盖）
- P1-04 · ECDICT 词库导入脚本（考试标签筛选、words/mastery 批量导入、Typer CLI 与单测覆盖）
- P1-02 · 数据库初始化脚本（init_db 执行 schema、启用 WAL/外键、幂等单测覆盖）
- P1-01 · 词汇数据库 schema（words、mastery、word_occurrences、collocations，约束与索引单测覆盖）
- P0-10 · P0 端到端 smoke test（NLP lemma 提取、DeepSeek JSON 小诗、Bark 推送串联验证）
- P0-09 · Tesseract OCR 封装（ocr_image / ocr_images，Pillow 预处理，段落感知文本清理，mock 单测覆盖）
- P0-08 · spaCy NLP 封装（tokenize_with_lemma / split_sentences / extract_unique_lemmas，lru_cache 单例，mock 单测覆盖）
- P0-07 · Bark 推送封装（push_bark，环境变量配置，mock 单测覆盖成功与失败路径）
- P0-06 · DeepSeek LLM 客户端封装（chat / chat_structured / embed，lru_cache 单例，100% 单测覆盖）
- P0-03 · Install spaCy + en_core_web_lg (~750MB English model)
- P0-04 · Install Tesseract OCR (system) + pytesseract + Pillow
- P0-05 · Configure DeepSeek API key in config/secrets.env
- P0-05 · Configure API keys (DeepSeek, Bark, SiliconFlow) and verify connectivity

## 2026-06-03
- P0-01 · Initialize project skeleton
- P0-02 · Install core dependencies (openai, fastapi, typer, ruff, pytest 等共 ~15 个直接依赖)
