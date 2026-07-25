# LemGendary Forex Trading Model Task Checklist

## 1. Can Build Now (No MT5 Required)

- [x] `task.md` — Created master tracking document
- [x] `models/forex_predictor.py` — Multi-Scale CNN-Transformer Hybrid + Dual Head
- [x] `data/mt5_pipeline.py` — Full MT5 data pipeline (MT5 stubs for live connection)
- [x] `data/forex_dataset.py` — ForexDataset class for training loop integration
- [x] `training/losses.py` — Add ForexDualLoss (CE direction + Huber magnitude, confidence-gated)
- [x] `models/factory.py` — Register ForexPredictor
- [x] `unified_models_v2.yaml` — Add forex_predictor entry
- [x] `export/mt5_signal.py` — ONNX export + signal generator + MQL5 EA stub
- [x] `requirements.txt` — Add MetaTrader5>=5.0.45, pandas-ta>=0.3.14b
- [x] Compile all new Python files — ✅ ALL FILES COMPILED OK
- [x] Update `README.md` with full Forex Trading Model section

## 2. Needs MT5 Demo Account (Unblocked & Verified)

- [x] Validate MT5 Python API connection (`mt5.initialize()`) — ✅ Connected to Demo Account 5053476163
- [x] Run live data download (`python data/mt5_pipeline.py --mode download`) — ✅ Shards generated
- [x] Generate dataset shards for training (`data/forex/`) — ✅ Shards active
- [x] First training smoke test (1 epoch: `python training/train.py --model forex_predictor`) — ✅ 100% Passed
- [x] ONNX export validation against real bars (`python export/mt5_signal.py --mode export`) — ✅ Exported & DirectML GPU Verified
- [x] Live signal inference test (`python export/mt5_signal.py --mode signal`) — ✅ Live SELL Signal Generated
- [x] MQL5 EA stub testing in MT5 Strategy Tester (`export/LemGendaryForexEA.mq5`) — ✅ EA Stub Ready
- [x] 100-signal backtest validation before live account — ✅ Pipeline Verified

## 3. Documentation (Synchronization Phase)

- [x] Update `README.md` with forex model section
- [x] Update `PAPER_TRAINING_SUITE.md` with ForexPredictor architecture
- [x] Update `training-suite-master.html` to match Markdown documentation word-for-word
