
## アプリの概要
・機械学習モデルを気軽に使用するためのアプリケーションの実装  
・モデルを作成し、外部のサーバに上げることで、外部の端末からでもモデルの使用を可能にする

## ファイル構成
ml_predict_app/  
|-app.py(アプリの動作管理)  
|-models/(モデルの格納フォルダ)  
|-image/(画像格納用フォルダ)  
|-requirements.txt(python環境でモデルを動かす際に使用)  
|-transforms.py(入力データの変換)  
|-templates/  
　|-start.html(一番最初のページ)  
　|-detail.html(モデルの説明ページ)  
　|-nokoshima.html(モデルを実際に使用するページ)  
