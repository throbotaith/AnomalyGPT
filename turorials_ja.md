# AnomalyGPT チュートリアル（日本語）

このドキュメントでは、リポジトリのクローンから Docker 上でデモを実行するまでの手順を説明します。

## 1. 環境準備

1. GitHub からリポジトリを取得します。
   ```bash
git clone https://github.com/CASIA-IVA-Lab/AnomalyGPT.git
cd AnomalyGPT
   ```
2. 依存パッケージをインストールします。
   ```bash
pip install -r requirements.txt
   ```

## 2. 必要なチェックポイントの配置

以下のファイルをそれぞれのディレクトリに配置してください。ディレクトリが存在しない場合は作成します。

1. **ImageBind** のチェックポイント `imagebind_huge.pth`
   - 取得先: [Facebook AI](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth)
   - 配置先: `pretrained_ckpt/imagebind_ckpt/`
2. **Vicuna** モデル
   - LLaMA 本体と Vicuna のデルタ重みを入手し、[pretrained_ckpt/README.md](./pretrained_ckpt/README.md) の手順に従って統合します。
   - 完成した重みを `pretrained_ckpt/vicuna_ckpt/` 以下に配置します。
3. **PandaGPT** のデルタ重み (pytorch_model.pt)
   - 取得先は README の表を参照してください。
   - 配置先: `pretrained_ckpt/pandagpt_ckpt/7b/` または `pretrained_ckpt/pandagpt_ckpt/13b/`
4. **AnomalyGPT** の重み
   - [Hugging Face](https://huggingface.co/FantasticGNU/AnomalyGPT) から取得します。
   - 配置先: `code/ckpt/`

## 3. Docker イメージのビルド

チェックポイントを配置したら、リポジトリのルートディレクトリで次のコマンドを実行して Docker イメージを作成します。
```bash
docker build -t anomalygpt .
```

## 4. Docker コンテナの起動

GPU を利用する場合の例を示します。カレントディレクトリをマウントし、ポート `7860` を公開します。
```bash
docker run --rm -it --gpus all -p 7860:7860 \
    -v $(pwd)/pretrained_ckpt:/app/pretrained_ckpt \
    -v $(pwd)/code/ckpt:/app/code/ckpt \
    -v $(pwd)/data:/app/data \
    anomalygpt
```
上記を自動化した `docker_run.sh` も用意されています。
```bash
bash docker_run.sh
```

## 5. デモの利用

コンテナが起動すると `web_demo.py` が実行され、Gradio サーバーが立ち上がります。ブラウザで次の URL にアクセスしてください。
```
http://localhost:7860
```
画面が表示されたら、アップロードした画像に対する異常検知を試すことができます。終了する場合はコンテナ内のプロセスを `Ctrl+C` で停止してください。

以上で Docker 上でのデモ実行までの流れは完了です。
