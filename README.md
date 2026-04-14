# 文化的背景と時空間制約を両立する高品質データセットを用いた翻訳モデルの構築
# Translation Modeling with High-Quality Datasets Integrating Cultural Context and Spatio-Temporal Constraints

Undergraduate thesis project, also presented at [情報処理学会自然言語処理研究会] (2026)


## 1. 研究概要 (Abstract)
動画配信サービスの普及により，字幕翻訳は単なる娯楽的補助から，言語横断的な情報流通を支える基盤技術としての重要性を増している．しかし，既存のニューラル機械翻訳モデルは原文忠実性を中心に設計されており，字幕翻訳特有の厳しい時間的・空間的制約の下で求められる高い情報圧縮性と文化的意味の再構成を同時に実現するには限界がある．本研究では，翻訳理論における機能主義的視点に基づき，人手による参照字幕と機械翻訳出力との乖離に着目したHard Sample Miningと，大規模言語モデルの専門知を活用するExpert Distillation を提案する．これにより，情報圧縮と文化的意味再構成の両立を可能とする字幕翻訳モデルの構築を目指す．評価実験では，自動指標および推論速度に基づく定量的分析に加え，人手評価による主観的品質判断，さらに推論過程におけるモデル内部の不確実性および確信度の動態解析を含む多面的な検証を行った．その結果，日英・英日の双方向字幕翻訳において，質的に選別された高品質学習データが翻訳性能を向上させることを確認した．また，高品質データは推論時に触媒的に機能し，デコーダの探索挙動を安定化させることで，翻訳品質と推論効率を同時に最適化できることが明らかとなった.

With the widespread adoption of video streaming services, subtitle translation has become a key technology for cross-lingual information dissemination. However, existing neural machine translation models, which prioritize source-text fidelity, are limited under the strict temporal and spatial constraints of subtitlesthat require both information compression and culturally appropriate meaning reconstruction. In this work, Motivated by the functionalist paradigm in translation theory, we propose a Hard Sample Mining approach focusing on discrepancies between human-authored subtitles and machine translations, along with an Expert Distillation framework that leverages specialized knowledge from large language models, to balance information compression and semantic reconfiguration. We conduct a comprehensive evaluation involving automatic
metrics, inference speed, human evaluation, and an analysis of model-internal uncertainty dynamics during inference. Experimental results on bidirectional Japanese-English and English-Japanese subtitle translation show that qualitatively curated high-quality training data improve translation performance, stabilize decoder search behavior, and enable simultaneous gains in translation quality and inference efficiency.


## 2. ディレクトリ構成 (Directory Structure)
本プロジェクトは，日英 (ja-en) および 英日 (en-ja) の双方向モデルを含みます．
プロジェクト全体の構成は以下の通りです (Total: 25 directories, 66 files)．

> **Note:** 視認性を高めるため，以下のツリー図では重複するファイル名（`MODEL1`〜`3`など）や同一構造のディレクトリを一部省略して記載しています．

```text
.
├── data/                                    # データは含まれません。OpenSubtitles-v2024 を各自取得してください
├── models/                                  # 学習済みモデル保存先（空またはreadme.txtのみ）
├── outputs/                                 # 出力結果保存先（空またはreadme.txtのみ）
├── README.md　
├── requirements.txt
|
└── src/                                     # ソースコード一式
    ├── 00_preprocess/                       # 前処理
    │   ├── 01_data_cleaning.ipynb
    │   ├── 07_create_all_dataset.py
    │   └── ...（その他整形スクリプト）
    |
    ├── 01_src_ja_en/                        # 【日英】実験コード
    │   ├── 01_train/                        # 学習
    │   │   ├── 01_parameter_optimization.py
    │   │   └── 02...04_model_train_MODEL[1-3].py
    |   |
    │   └── 02_evaluation/                   # 評価・分析
    │       ├── 00_prepare_of_models_output/ # モデル別出力結果の準備
    │       │   ├── 01_distribute_japanese.py
    │       │   ├── 02_basemodel_output_text.py
    │       │   └── 03...05_model[1-3]_output_text.py
    |       |
    │       ├── 01_translation_quality/      # 翻訳品質(bleu,comet,bert)
    │       │   └── 01_final_evaluation_ALL_MODELS.py
    |       |
    │       ├── 02_evaluate_output/          # 確信度・生成長
    │       │   ├── 01_basemodel_confidence_entrophy.py
    │       │   ├── 02...04_model[1-3]_confidence_entrophy.py
    │       │   ├── 05_grapth_confidence_entrophy.py
    │       │   └── 06_lengh_of generation.py
    |       |
    │       └── 03_prepare_of_human_eval/    # 人手評価のデータ準備
    │           └── 01_add_row_models_transllation_ja_en.py
    |
    ├── 02_src_en_ja/　                      # 【英日】実験コード
    │   └── ...（構成は 01_src_ja_enと同一のため省略）
    │   
    ├── 90_misc/                             # グラフ描画・統計確認（適宜利用）
    │   ├── 01_file_word_error_check.py
    │   └── ...
    |
    └── 99_future_work/                      # 今後の展望・実験コード
        ├── en_ja/
        │   ├── 01_multi_agent.py
        │   └── ...
        └── ja_en/
            ├── 01_mt_test.py
            └── ...
```

## 3. 動作環境 (Requirements)
本コードは共有サーバ（Linux / NVIDIA RTX 4000 Ada Generation環境）にて動作確認を行っています.
* **Python:** 3
* **主要ライブラリ:** PyTorch, Transformers...
* **その他:** 詳細は同梱の `requirements.txt` を参照してください．


## 4. セットアップ手順 (Setup)
以下のコマンドを実行し，必要なパッケージをインストールしてください．

```bash
pip install -r requirements.txt
```


## 5. 実行方法 (Usage)
各ディレクトリ（`01_train`, `02_evaluation` 等）内のスクリプトを，**ファイル名の番号順（01_..., 02_...）** に実行してください．

> **⚠️ 注意 (Note):**
> コード内のファイルパスは著者の環境に合わせて記述されています．
> 実行時は，各スクリプト冒頭の `DATA_PATH` や `MODEL_DIR` をご自身の環境に合わせて書き換えてください．
> データの前処理については論文（Thesis）の記述を参照してください．`src/00_preprocess` に格納されていますが，実行環境に依存する部分が含まれます．

1. **前処理 (Preprocessing):** `src/00_preprocess` 内のスクリプトを番号順に実行
2. **学習 (Training):** `src/01_src_ja_en/01_train` および`src/02_src_en_ja/01_train` 内のスクリプトを番号順に実行
3. **評価 (Evaluation):** `src/01_src_ja_en/02_evaluation` および`src/02_src_en_ja/02_evaluation` 内のスクリプトを番号順に実行

**▼ 日英方向の評価コードの実行例 (Example: Evaluation)**
```bash
# 1. 推論結果の出力・整理
cd src/01_src_ja_en/02_evaluation/00_prepare_of_models_output
python3 01_distribute_japanese.py
python3 02_basemodel_output_text.py
# ... (番号順に実行)

# 2. 翻訳品質の評価
cd ../01_translation_quality
python3 01_final_evaluation_ALL_MODELS.py

# 3. エントロピー・確信度の分析
cd ../02_evaluate_output
python3 01_basemodel_confidence_entrophy.py
# ... (番号順に実行)
```


## 6. 特記事項 (Notes)
* **コードの解説について:** 実装の詳細や各関数の説明については，各ソースコード内にコメントとして記述しています．詳細な処理内容を確認する際は，該当ファイルを直接参照してください．
* **データの取得について:** 本リポジトリにはデータを含めていません OpenSubtitles-v2024から取得してください．
* **パス設定について：** コード内のファイルパスについて，実行する際は，ご自身の環境に合わせて **パス設定を書き換えてから** 実行してください．
* **実行環境について：** 学習時のバッチサイズや推論速度は，実行環境のGPUスペックに依存します

## 7.ライセンス (License)
本リポジトリのコード部分は MIT License の下で公開しています．本研究で使用したデータセットおよび事前学習モデルの利用については，各提供元のライセンス・利用規約をご確認ください．

## 8. 参考文献 (references)
以下は，本研究に関する卒業論文および自然言語処理研究会（NL研究会）発表論文に記載した参考文献です．

[1] 藤濤文子：翻訳行為と異文化間コミュニケーション：機能主義的翻訳理論の諸相，松籟社(2007).
[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,Jones, L., Gomez, A. N., Kaiser, L. u. and Polosukhin, I.: Attention is All you Need, Advances in Neural Information Processing Systems (Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R.,Vishwanathan, S. and Garnett, R., eds.), Vol. 30, Curran Associates, Inc., (online),available from ⟨https://proceedings.neurips.cc/paper files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf⟩ (2017).
[3] Baker, M. and Saldanha, G.(eds.): 翻訳研究のキーワード，研究社(2013). 藤濤文子 監修・編訳,伊原紀子・田辺希久子訳.
[4] Karamitroglou, F.: A Proposed Set of Subtitling Standards in Europe, Translation Journal, Vol. 2, No. 2, pp.1–15 (online), available from ⟨https://translationjournal.net/journal/04stndrd.htm⟩ (1998).
[5] 保坂敏子：字幕翻訳で失われる要素：言語教育との関わりを考える，日本語と日本語教育， No. 44, pp. 41–57（オンライン），入手先⟨https://koara.lib.keio.ac.jp/xoonips/modules/xoonips/detail.php?koara id=AN00189695-20160300-0041⟩(2016).
[6] De Linde, Z. and Kay, N.: The Semiotics of Subtitling, St. Jerome (1999).
[7] Freitag, M. and Al-Onaizan, Y.: Fast Domain Adaptation for Neural Machine Translation (2016).
[8] Hinton, G., Vinyals, O. and Dean, J.: Distilling the Knowledge in a Neural Network (2015).
[9] Kim, Y. and Rush, A. M.: Sequence-Level Knowledge Distillation, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (Su, J., Duh, K and Carreras, X., eds.), Austin, Texas, Association for Computational Linguistics, pp. 1317–1327 (online), DOI: 10.18653/v1/D16-1139 (2016).
[10] Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q. and Artzi, Y.: BERTScore: Evaluating Text Generation with BERT, 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020, OpenReview.net, (online), available from ⟨https://openreview.net/forum?id=SkeHuCVFDr⟩(2020).
[11] Lison, Pierre and Tiedemann, J¨org, editor = ”Calzolari, N., Choukri, K., Declerck, T., Goggi, S., Grobelnik, M., Maegaard, B., Mariani, J., Mazo, H., Moreno, A. Odijk, J. and Piperidis, S.: OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles, Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), Portoroˇz, Slovenia, European Language Resources Association (ELRA), pp. 923–929 (online), available from ⟨https://aclanthology.org/L16-1147/⟩ (2016).
[12] Tiedemann, J¨org and Thottingal, Santhosh: OPUS-MT– Building open translation services for the World, Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (Martins, A., Moniz, H., Fumega, S., Martins, B., Batista, F., Coheur, L., Parra, C., Trancoso, I., Turchi, M., Bisazza, A., Moorkens, J., Guerberof, A., Nurminen, M., Marg, L. and Forcada, M. L., eds.), Lisboa, Portugal, European Association for Machine Translation, pp. 479–480(online), available from ⟨https://aclanthology.org/2020.eamt-1.61/⟩ (2020).
[13] Helsinki-NLP: OPUS-MT Japanese–English Translation Model, https://huggingface.co/Helsinki-NLP/opus-mt-ja-en. Model distribution page on Hugging Face. Accessed: 2025-11.
[14] staka: FuguMT: English–Japanese Neural Machine Translation Model, https://staka.jp/wordpress/. Model available at https://huggingface.co/staka/fugumt-en-ja. Accessed: 2026-01.
[15] Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M.: Optuna: A next-generation hyperparameter optimization framework, Proceedings of the 25th ACMSIGKDDinternational conference on knowledge discovery & data mining, pp. 2623–2631 (2019).
[16] Loshchilov, I. and Hutter, F.: Decoupled Weight Decay Regularization, International Conference on Learning Representations, (online), available from ⟨https://openreview.net/forum?id=Bkg6RiCqY7⟩ (2019).
[17] Papineni, K., Roukos, S., Ward, T. and Zhu, W.J.: Bleu: a Method for Automatic Evaluation of Machine Translation, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (Isabelle, P., Charniak, E. and Lin, D., eds.), Philadelphia, Pennsylvania, USA, Association for Computational Linguistics, pp 311–318 (online), DOI:10.3115/1073083.1073135 (2002).
[18] Post, M.: A Call for Clarity in Reporting BLEU Scores, Proceedings of the Third Conference on Machine Translation: Research Papers (Bojar, O., Chatterjee, R., Fed
ermann, C., Fishel, M., Graham, Y., Haddow, B., Huck, M., Yepes, A. J., Koehn, P., Monz, C., Negri, M., N´ev´eol, A., Neves, M., Post, M., Specia, L., Turchi, M. and
Verspoor, K., eds.), Brussels, Belgium, Association for Computational Linguistics, pp. 186–191 (online), DOI: 10.18653/v1/W18-6319 (2018).
[19] Rei, R., Stewart, C., Farinha, A. C. and Lavie, A.: COMET: ANeural Framework for MT Evaluation, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (Webber, B., Cohn, T., He, Y. and Liu, Y., eds.), Online, Association for Computational Linguistics, pp. 2685–2702 (online), DOI: 10.18653/v1/2020.emnlp-main.213 (2020).
[20] Blatz, J., Fitzgerald, E., Foster, G., Gandrabur, S., Goutte, C., Kulesza, A., Sanchis, A. and Ueffing, N.: Confidence Estimation for Machine Translation, COL ING 2004: Proceedings of the 20th International Conference on Computational Linguistics, Geneva, Switzerland, COLING, pp. 315–321 (online), available from ⟨https://aclanthology.org/C04-1046/⟩ (2004).
[21] Shannon, C. E.: A Mathematical Theory of Communication, The Bell System Technical Journal, Vol. 27, No. 3, pp. 379–423 (online), DOI: 10.1002/j.15387305.1948.tb01338.x (1948)


## 9. 謝辞 (Acknowledgments)
本研究の遂行にあたり，指導教員の先生をはじめとする研究室の皆様，ならびに評価実験にご協力いただいた留学生・学生の皆様に深く感謝いたします．
