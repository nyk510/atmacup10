# atmaCup#10 vivid model

## Requirement

* docker
* docker-compose

## setup

```bash
cp project.env .env
docker-compose up -d --build

# python file を実行するときには docker の内部で
docker exec -it atmacup-10-jupyter  bash
```

### Note

* data を保存するディレクトリを `.env` から変更できます (DATA_DIRをデータの入っているディレクトリに変更してください。デフォルトは `./data` になっています)
* `.env` は `docker-compose up` のタイミングでしか更新されません。編集をした場合は `docker-compose up` してください。

## run experiment

run `src/exp__027.py` in docker container.

```bash
penguin@9e5324b534a4:/analysis$ python src/exp__027.py --h
usage: exp__027.py [-h] [--input INPUT] [--output OUTPUT] [--force] [--simple]

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    input directory (default: /analysis/data/inputs)
  --output OUTPUT  output directory (default: /analysis/data/outputs)
  --force          If add me, re-create all models. (default: False)
  --simple         if add me, create lightgbm model only. (skip other models)
                   (default: False)
```

## Solution outline

### 環境

```dockerfile
FROM registry.gitlab.com/nyker510/analysis-template/cpu:1.0.4

RUN pip install -U pip && \
  pip install \
    python-vivid==0.3.3.4 \
    shortuuid \
    interpret \
    pygam \
    dataclasses_json \
    texthero \
    pip install git+https://gitlab+deploy-token-373496:JZxuUxVmg682HGji1Zfs@gitlab.com/atma_inc/anemone.git \
    pandas==1.2.2

WORKDIR /home/penguin
RUN git clone https://github.com/facebookresearch/fastText.git && \
  cd fastText && \
  pip install . && \
  rm -rf /home/penguin/fastText
```

### 特徴量

名前とコード内の block 名の順で記載しています。

#### 一般的なもの

* CountEncoding
* TargetEncoding
* 文字列の長さ
* 作家の年齢や制作年度の差分等の年度情報 (MakerYaerBlock)

#### 色情報

* color で一番 percentage の大きい色の rgb (MainColorBlock)
* rgb 意外の hsv / hls / yiq の情報 (OtherMainColorBlock / ConvertedMainColorBlock)
  * importance では割と上位に位置していた

#### テキスト系特徴量

* sub_title に含まれる属性情報をパースして with / height などに展開 (ArtAttributeBlock)
* テキスト系カラムを Tfidf (TfidfBlock)
  * tfidf -> truncated svd で圧縮
  * 正規化の際にはオランダ語の stopword も含める
* Fasttext による埋め込みを使った SWEM (FasttextEmbeddingBlock)
  * オランダ語の pretrained model を利用
  * embedding は 100 次元に圧縮
  * 今見たら stopword の除去やってないことに気が付きました。

#### 集約系

* 外部に紐づくテーブルの `name` を one-hot encoding (One2ManyBlock)
* 外部に紐づくテーブルのレコード数 (RelationCountBlock)
* material の名前ごとに年度のランク付け (DatingRankByMaterialBlock)
* 何年間かけて作成してるか (PaintingDurationBlock)
* 紐づく principal maker で出現する qualification / roles / productionPlaces の one-hot
  * カウントベース: PrinciapMakerCountBlock
  * カウントしたあとに全出現回数で正規化 (PrincipalMakerMetaBlock)
* 全 art_object で principal maker ごとの他のカラムのユニーク数・割合 (PrincipalMakerOtherFieldMetaBlock)
* 作家ごとに principal maker の情報を集約した特徴 (MakerCountBlock)

#### その他

* 作家の誕生日が抜けているのを sorting date の最小値で穴埋めする (FilledMakerBirthBlock)
* 制作日と取得日の差分 (AcquisitionAndDatingBlock)

### モデル

* single
  * lightgbm x5 (objective=rmse / poisson)
  * xgboost x5
  * random forest
  * ridge
  * svm (Bagging 利用)
* stacking
  * ridge
  * lightgbm
  * もとの特徴量と single model を混ぜた lightgbm

結局 lightgbm x5 が CV / LB ともに stakcing と同程度だったのでモデル部分はあまり多様性に寄与していないと思います。
