
# Neural SDF サンプル

このリポジトリは、以下の2段構成でSDFを試せます。

- `sdf_primitives.py`: 通常のSDF（球・ボックス）
- `neural_sdf_train.py`: Neural SDF学習 + Eikonal loss + Marching Cubesメッシュ化

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. 通常SDFのテスト

```bash
python3 sdf_primitives.py
```

## 2. Neural SDF学習（Eikonal lossあり）

```bash
python3 neural_sdf_train.py
```

形状を切り替える例:

```bash
python3 neural_sdf_train.py --shape sphere
python3 neural_sdf_train.py --shape box
python3 neural_sdf_train.py --shape composite
```

実行すると以下が生成されます（shapeごと）。

- `generated/sdf_<shape>_model.pt`: 学習済み重み
- `generated/mesh_<shape>.obj`: Marching Cubesで抽出した0レベルセットのメッシュ

## メモ

- Eikonal項は `|grad f|-1` を最小化し、SDFらしい勾配場を維持して表面品質を改善します。
- メッシュ抽出は `scikit-image` の `measure.marching_cubes` を使用しています。
- `composite` は球とボックスの合成（union）に穴あけ（difference）を入れた複合形状です。



これを参考にしてる
https://chatgpt.com/share/69d374a4-0498-83e8-a280-45436487a289