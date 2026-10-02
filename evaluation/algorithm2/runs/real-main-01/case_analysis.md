# 失败案例分析（真实测试 query）

运行 `real-main-01`（级别 main）。案例全部取自 predictions.parquet 的测试 query，按 query_id 排序后在每类中选取，不挑选“好看”的例子。核心模型：A0、A1 与部署 seed 42 的 B0 / B1 / C0。

阅读说明：**观察**是从数据直接读出的事实；**待验证推测**只是可能原因，未经对照实验验证，特征重要性与案例相关性都不是因果解释。标签是代理标签（评分 ≥ 8），未评分的番剧是弱负例，“未命中”不代表用户不喜欢推荐结果。

| 类别 | 测试中出现次数 | 展示案例数 |
|---|---|---|
| A0 命中、学习模型（B1 与 C0）未命中 | 26 | 2 |
| 学习模型命中、A0 未命中 | 518 | 2 |
| 五个核心模型都未命中 | 2399 | 2 |
| 稀疏收藏（可见历史 1–4 部） | 600 | 2 |
| 混合偏好（genre 熵高于验证集中位数） | 1809 | 2 |
| 规则排除了隐藏目标（严格模式不插入） | 5146 | 3 |
| 严格空结果（候选数 = 0） | 0 | 0 |
| 候选不足 8 个 | 0 | 0 |

## A0 命中、学习模型（B1 与 C0）未命中

测试集中符合条件的 query 数：26。
### 案例 1：A0 命中、学习模型（B1 与 C0）未命中 · query `test-1088591-mal:1840`

- 离线用户 1088591 · 主组 · 可见收藏 77 部 · 规则 R0（无筛选）
- 可见收藏：Naruto；School Rumble；Neon Genesis Evangelion；.hack//Sign；DearS；Black Cat (TV)；Shuffle!；Air；Fullmetal Alchemist；Vandread；…（共 77 部）
- 画像 top-10 标签：genre:Comedy, theme:School, genre:Action, genre:Romance, genre:Fantasy, genre:Ecchi, genre:Sci-Fi, genre:Adventure, genre:Drama, genre:Supernatural
- 隐藏目标：**Zero no Tsukaima: Futatsuki no Kishi**（mal:1840）· 公开评分 7.41 · 集数 12 · 人气排名 415/14182 · 标签 genre:Action, genre:Adventure, genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:Isekai, theme:Military, theme:School · 制作公司 J.C.Staff
- 目标与画像 top-10 标签重叠 7 个（genre:Action, genre:Adventure, genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:School）；目标的制作公司出现在可见收藏中
- 候选数 14105；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2 | 1 | Urusei Yatsura / Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Urusei Yatsura Movie 1:… / Photon / Blue Seed / Gakusen Toshi Asterisk … / Negima!? |
| A1 | 4 | 1 | Urusei Yatsura / Cardcaptor Sakura / Gintama: The Final / Zero no Tsukaima: Futat… / Kaguya-sama wa Kokurase… / Urusei Yatsura 2: Beaut… / Zero no Tsukaima: Princ… / Tsubasa: Tokyo Revelati… |
| B0_seed42 | 9 | 0 | One Piece / Mononoke Hime / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… / Tenkuu no Shiro Laputa / Cardcaptor Sakura / Zero no Tsukaima: Princ… / Clannad: After Story |
| B1_seed42 | 9 | 0 | One Piece / Zero no Tsukaima: Princ… / Rosario to Vampire Capu2 / Clannad: After Story / Bleach / Prison School / Gintama' / Gintama': Enchousen |
| C0_seed42 | 74 | 0 | Death Note / Ouran Koukou Host Club / Clannad: After Story / Clannad / One Piece / Kaichou wa Maid-sama! / Katekyo Hitman Reborn! / Angel Beats! |

**观察**：A0 的 Top-8 平均公开评分 7.19、平均 hits 7.0；目标公开评分 7.41、hits 7。学习模型对目标的名次：B0_seed42 9，B1_seed42 9，C0_seed42 74。

**待验证推测**：目标与画像标签高度重叠（hits 7，不低于 A0 推荐条目的平均值），A0 的 hits 项把它排在前面；学习模型更依赖公开评分（B1 中 rating_norm 的 gain 占比 66%），把评分或人气更高的条目排在它前面。

### 案例 2：A0 命中、学习模型（B1 与 C0）未命中 · query `test-1103683-mal:1313`

- 离线用户 1103683 · 主组 · 可见收藏 35 部 · 规则 R0（无筛选）
- 可见收藏：Gakuen Alice；Full Moon wo Sagashite；Yumeria；Cardcaptor Sakura；Puchi Pri*Yucie；Psychic Academy；Cardcaptor Sakura Movie 1；Cardcaptor Sakura Movie 2: Fuuin Sareta…；Yu☆Gi☆Oh! Duel Monsters；Yu☆Gi☆Oh! Duel Monsters GX；…（共 35 部）
- 画像 top-10 标签：genre:Comedy, genre:Fantasy, theme:School, genre:Romance, theme:Mahou Shoujo, genre:Adventure, genre:Drama, genre:Action, theme:Coming of Age, theme:Isekai
- 隐藏目标：**Digimon Adventure 02**（mal:1313）· 公开评分 7.23 · 集数 50 · 人气排名 1033/14182 · 标签 genre:Action, genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, theme:Coming of Age, theme:Isekai, theme:Mythology · 制作公司 Toei Animation
- 目标与画像 top-10 标签重叠 7 个（genre:Action, genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, theme:Coming of Age, theme:Isekai）；目标的制作公司出现在可见收藏中
- 候选数 14147；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 8 | 1 | Bishoujo Senshi Sailor … / Cardcaptor Sakura: Clea… / Flip Flappers / Sugar Sugar Rune / Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Digimon Adventure 02 |
| A1 | 21 | 0 | Bishoujo Senshi Sailor … / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Cardcaptor Sakura: Clea… / Flip Flappers / Princess Tutu / Fullmetal Alchemist: Br… / Tsubasa: Shunraiki |
| B0_seed42 | 9 | 0 | Fullmetal Alchemist: Br… / Death Note / Fullmetal Alchemist / One Piece / Mononoke Hime / Kaguya-sama wa Kokurase… / Naruto: Shippuuden / Zero no Tsukaima |
| B1_seed42 | 23 | 0 | Ojamajo Doremi Dokkaan! / Ojamajo Doremi Na-i-sho / Ojamajo Doremi / Motto! Ojamajo Doremi / One Piece / Ojamajo Doremi Sharp / Digimon Xros Wars: Aku … / Lovely★Complex |
| C0_seed42 | 133 | 0 | Naruto / Lovely★Complex / Bishoujo Senshi Sailor … / Bleach / One Piece / Junjou Romantica 2 / Fruits Basket / Ouran Koukou Host Club |

**观察**：A0 的 Top-8 平均公开评分 7.49、平均 hits 7.0；目标公开评分 7.23、hits 7。学习模型对目标的名次：B0_seed42 9，B1_seed42 23，C0_seed42 133。

**待验证推测**：目标与画像标签高度重叠（hits 7，不低于 A0 推荐条目的平均值），A0 的 hits 项把它排在前面；学习模型更依赖公开评分（B1 中 rating_norm 的 gain 占比 66%），把评分或人气更高的条目排在它前面。

## 学习模型命中、A0 未命中

测试集中符合条件的 query 数：518。
### 案例 3：学习模型命中、A0 未命中 · query `test-101298-mal:1575`

- 离线用户 101298 · 主组 · 可见收藏 6 部 · 规则 R0（无筛选）
- 可见收藏：Full Metal Panic!；Full Metal Panic? Fumoffu；Full Metal Panic! The Second Raid；Elfen Lied；Great Teacher Onizuka；Kiba
- 画像 top-10 标签：genre:Action, theme:Military, theme:School, genre:Comedy, theme:Delinquents, theme:Mecha, genre:Adventure, genre:Drama, genre:Fantasy, genre:Horror
- 隐藏目标：**Code Geass: Hangyaku no Lelouch**（mal:1575）· 公开评分 8.70 · 集数 25 · 人气排名 21/14182 · 标签 genre:Action, genre:Award Winning, genre:Drama, genre:Sci-Fi, theme:Mecha, theme:Military, theme:School, theme:Super Power · 制作公司 Sunrise
- 目标与画像 top-10 标签重叠 5 个（genre:Action, genre:Drama, theme:Mecha, theme:Military, theme:School）；目标的制作公司未出现在可见收藏中
- 候选数 14176；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 12 | 0 | Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Top wo Nerae! Gunbuster / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Urusei Yatsura / Berserk: Ougon Jidai-he… / Pokemon Movie 01: Mewtw… |
| A1 | 3 | 1 | Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / Code Geass: Hangyaku no… / Berserk: Ougon Jidai-he… / Code Geass: Hangyaku no… / Top wo Nerae! Gunbuster / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… |
| B0_seed42 | 2 | 1 | Fullmetal Alchemist: Br… / Code Geass: Hangyaku no… / Death Note / Fullmetal Alchemist / Kenpuu Denki Berserk / One Piece / Mononoke Hime / Code Geass: Hangyaku no… |
| B1_seed42 | 7 | 1 | Fullmetal Alchemist: Br… / Death Note / One Piece / Tengen Toppa Gurren Lag… / Violet Evergarden / Suzumiya Haruhi no Shou… / Code Geass: Hangyaku no… / Naruto: Shippuuden |
| C0_seed42 | 5 | 1 | Death Note / Fullmetal Alchemist / Fullmetal Alchemist: Br… / One Piece / Code Geass: Hangyaku no… / Naruto / Code Geass: Hangyaku no… / Naruto: Shippuuden |

**观察**：A0 的 Top-8 平均公开评分 7.93、平均 hits 6.0；目标公开评分 8.70、hits 5。学习模型对目标的名次：B0_seed42 2，B1_seed42 7，C0_seed42 5。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。B1 / C0 使用了制作公司、集数、相似度等 A0 没有的信号，可能因此排得更前；这只是相关观察，不是因果证明。

### 案例 4：学习模型命中、A0 未命中 · query `test-10152-mal:164`

- 离线用户 10152 · 主组 · 可见收藏 22 部 · 规则 R0（无筛选）
- 可见收藏：Naruto；.hack//Sign；Aa! Megami-sama! (TV)；Ai Yori Aoshi；D.N.Angel；Full Metal Panic!；Mai-HiME；Ayashi no Ceres；Ou Dorobou Jing；Onegai☆Teacher；…（共 22 部）
- 画像 top-10 标签：genre:Adventure, genre:Fantasy, genre:Action, genre:Comedy, genre:Romance, theme:School, genre:Drama, genre:Supernatural, genre:Award Winning, theme:Coming of Age
- 隐藏目标：**Mononoke Hime**（mal:164）· 公开评分 8.67 · 集数 1 · 人气排名 105/14182 · 标签 genre:Action, genre:Adventure, genre:Award Winning, genre:Fantasy · 制作公司 Studio Ghibli
- 目标与画像 top-10 标签重叠 4 个（genre:Action, genre:Adventure, genre:Award Winning, genre:Fantasy）；目标的制作公司出现在可见收藏中
- 候选数 14160；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 136 | 0 | Sennen Joyuu / Cardcaptor Sakura / Nakitai Watashi wa Neko… / Urusei Yatsura Movie 1:… / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Little Busters! Refrain / Fullmetal Alchemist |
| A1 | 26 | 0 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Little Busters! Refrain / Fullmetal Alchemist / Fullmetal Alchemist: Br… / Kurenai no Buta |
| B0_seed42 | 3 | 1 | Fullmetal Alchemist: Br… / Fullmetal Alchemist / Mononoke Hime / Toki wo Kakeru Shoujo / Code Geass: Hangyaku no… / One Piece / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… |
| B1_seed42 | 1 | 1 | Mononoke Hime / Fullmetal Alchemist: Br… / Kono Subarashii Sekai n… / One Piece / Code Geass: Hangyaku no… / Juuni Kokuki / 86 Part 2 / Akatsuki no Yona |
| C0_seed42 | 32 | 0 | Death Note / One Piece / Fullmetal Alchemist: Br… / Fullmetal Alchemist / Bleach: Sennen Kessen-h… / Code Geass: Hangyaku no… / Code Geass: Hangyaku no… / Ouran Koukou Host Club |

**观察**：A0 的 Top-8 平均公开评分 7.94、平均 hits 6.5；目标公开评分 8.67、hits 4。学习模型对目标的名次：B0_seed42 3，B1_seed42 1，C0_seed42 32。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。B1 / C0 使用了制作公司、集数、相似度等 A0 没有的信号（目标的制作公司出现在可见收藏中），可能因此排得更前；这只是相关观察，不是因果证明。

## 五个核心模型都未命中

测试集中符合条件的 query 数：2399。
### 案例 5：五个核心模型都未命中 · query `test-1000595-mal:241`

- 离线用户 1000595 · 主组 · 可见收藏 72 部 · 规则 R0（无筛选）
- 可见收藏：Chrno Crusade；Air；Final Approach；Fullmetal Alchemist；Green Green；InuYasha；Happy☆Lesson；Happy☆Lesson (TV)；Happy☆Lesson Advance；Bleach；…（共 72 部）
- 画像 top-10 标签：theme:School, genre:Romance, genre:Comedy, genre:Action, genre:Drama, genre:Ecchi, genre:Sci-Fi, genre:Supernatural, genre:Fantasy, theme:Coming of Age
- 隐藏目标：**Girls Bravo: First Season**（mal:241）· 公开评分 6.38 · 集数 11 · 人气排名 1457/14182 · 标签 genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:Isekai, theme:School · 制作公司 AIC Spirits
- 目标与画像 top-10 标签重叠 5 个（genre:Comedy, genre:Ecchi, genre:Fantasy, genre:Romance, theme:School）；目标的制作公司未出现在可见收藏中
- 候选数 14110；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 292 | 0 | Chobits / Urusei Yatsura Movie 1:… / Blue Seed / Gakusen Toshi Asterisk … / Gakusen Toshi Asterisk / Little Busters! Refrain / Cardcaptor Sakura / Top wo Nerae! Gunbuster |
| A1 | 1513 | 0 | Little Busters! Refrain / Cardcaptor Sakura / Gintama: The Final / Chobits / Kaguya-sama wa Kokurase… / Top wo Nerae! Gunbuster / Bakuman. 2nd Season / Shinsekai yori |
| B0_seed42 | 746 | 0 | Toki wo Kakeru Shoujo / Chobits / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… / Cardcaptor Sakura / Clannad: After Story / Bakuman. 2nd Season / Mushoku Tensei: Isekai … |
| B1_seed42 | 667 | 0 | Yahari Ore no Seishun L… / High School DxD New / Yahari Ore no Seishun L… / Clannad: After Story / One Piece / Dr. Stone: New World / Prison School / InuYasha: Kanketsu-hen |
| C0_seed42 | 257 | 0 | Ouran Koukou Host Club / Clannad / Clannad: After Story / Lucky☆Star / Death Note / Durarara!! / Kanon (2006) / K-On! |

**观察**：A0 的 Top-8 平均公开评分 7.44、平均 hits 6.6；目标公开评分 6.38、hits 5。学习模型对目标的名次：B0_seed42 746，B1_seed42 667，C0_seed42 257。

**待验证推测**：目标的公开评分明显低于 A0 推荐条目的平均值；以公开评分为主要信号的排序可能把它压到后面。目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。

### 案例 6：五个核心模型都未命中 · query `test-100433-mal:401`

- 离线用户 100433 · 主组 · 可见收藏 9 部 · 规则 R0（无筛选）
- 可见收藏：Trigun；Naruto；School Rumble；Rurouni Kenshin: Meiji Kenkaku Romantan…；Rurouni Kenshin: Meiji Kenkaku Romantan；Rurouni Kenshin: Meiji Kenkaku Romantan…；Kidou Senshi Gundam SEED；Bleach；Naruto: Shippuuden
- 画像 top-10 标签：genre:Action, genre:Adventure, genre:Romance, genre:Drama, genre:Fantasy, theme:Historical, theme:Martial Arts, theme:Samurai, genre:Comedy, genre:Sci-Fi
- 隐藏目标：**Rurouni Kenshin: Meiji Kenkaku Romantan - Seisou-hen**（mal:401）· 公开评分 7.97 · 集数 2 · 人气排名 2160/14182 · 标签 genre:Drama, genre:Romance, theme:Historical, theme:Samurai · 制作公司 Studio Deen
- 目标与画像 top-10 标签重叠 4 个（genre:Drama, genre:Romance, theme:Historical, theme:Samurai）；目标的制作公司出现在可见收藏中
- 候选数 14173；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 277 | 0 | Gintama: The Final / Saber Marionette J / Katanagatari / Jigokuraku / Sennen Joyuu / Cardcaptor Sakura / Vampire Hunter / Kurenai no Buta |
| A1 | 174 | 0 | Gintama: The Final / Gintama° / Gintama' / Gintama': Enchousen / Gintama. / Gintama / Gintama Movie 2: Kanket… / Gintama.: Shirogane no … |
| B0_seed42 | 227 | 0 | Fullmetal Alchemist: Br… / Jigokuraku / Gintama Movie 2: Kanket… / Gintama.: Shirogane no … / Death Note / The First Slam Dunk / Samurai Champloo / Stranger: Mukou Hadan |
| B1_seed42 | 18 | 0 | Fullmetal Alchemist: Br… / Death Note / One Piece / Bleach: Sennen Kessen-h… / Samurai Champloo / Trigun: Badlands Rumble / Code Geass: Hangyaku no… / Cowboy Bebop |
| C0_seed42 | 230 | 0 | One Piece / Fullmetal Alchemist / Code Geass: Hangyaku no… / Fullmetal Alchemist: Br… / Death Note / Bleach: Sennen Kessen-h… / Code Geass: Hangyaku no… / Dragon Ball Z |

**观察**：A0 的 Top-8 平均公开评分 7.99、平均 hits 6.2；目标公开评分 7.97、hits 4。学习模型对目标的名次：B0_seed42 227，B1_seed42 18，C0_seed42 230。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。目标在目录中较冷门（人气排名 2160），而各模型的主要信号与公开评分 / 人气相关，冷门目标普遍难以进入前 8。

## 稀疏收藏（可见历史 1–4 部）

测试集中符合条件的 query 数：600。
### 案例 7：稀疏收藏（可见历史 1–4 部） · query `test-100898-mal:63`

- 离线用户 100898 · 稀疏组 · 可见收藏 3 部 · 规则 R0（无筛选）
- 可见收藏：Yu☆Gi☆Oh! Duel Monsters；Digimon Adventure；Death Note
- 画像 top-10 标签：genre:Action, genre:Adventure, genre:Fantasy, genre:Comedy, genre:Supernatural, genre:Suspense, theme:Coming of Age, theme:Historical, theme:Isekai, theme:Mythology
- 隐藏目标：**DearS**（mal:63）· 公开评分 6.56 · 集数 12 · 人气排名 1451/14182 · 标签 genre:Comedy, genre:Ecchi, genre:Romance, genre:Sci-Fi · 制作公司 Daume
- 目标与画像 top-10 标签重叠 1 个（genre:Comedy）；目标的制作公司未出现在可见收藏中
- 候选数 14179；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 7454 | 0 | Digimon Adventure 02 / Dragon Ball Z / Seirei no Moribito / Akatsuki no Yona / Fate/Grand Order: Zetta… / Drifters / Drifters: Special Editi… / Fushigi Yuugi |
| A1 | 7207 | 0 | Dragon Ball Z / Seirei no Moribito / Mo Dao Zu Shi: Wanjie P… / Akatsuki no Yona / Fate/Grand Order: Zetta… / Mo Dao Zu Shi / Mo Dao Zu Shi: Xian Yun… / Drifters |
| B0_seed42 | 6679 | 0 | Fullmetal Alchemist: Br… / Ping Pong the Animation / The First Slam Dunk / Akatsuki no Yona / One Piece / Mononoke Hime / Fate/stay night Movie: … / Naruto: Shippuuden |
| B1_seed42 | 3973 | 0 | One Piece / Dragon Ball Z / Hunter x Hunter (2011) / Dragon Ball / Dragon Ball Kai (2014) / Fullmetal Alchemist: Br… / Saint Seiya / Dragon Ball Kai |
| C0_seed42 | 583 | 0 | Naruto / One Piece / Fullmetal Alchemist / Naruto: Shippuuden / Fullmetal Alchemist: Br… / Bleach / Code Geass: Hangyaku no… / Bleach: Sennen Kessen-h… |

**观察**：A0 的 Top-8 平均公开评分 7.83、平均 hits 6.1；目标公开评分 6.56、hits 1。学习模型对目标的名次：B0_seed42 6679，B1_seed42 3973，C0_seed42 583。

**待验证推测**：目标的公开评分明显低于 A0 推荐条目的平均值；以公开评分为主要信号的排序可能把它压到后面。目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏只有 3 部，每部占画像频率的 1/3，标签信号噪声大。

### 案例 8：稀疏收藏（可见历史 1–4 部） · query `test-100963-mal:71`

- 离线用户 100963 · 稀疏组 · 可见收藏 3 部 · 规则 R0（无筛选）
- 可见收藏：D.Gray-man；Shijou Saikyou no Deshi Kenichi；Macross F
- 画像 top-10 标签：genre:Action, genre:Adventure, genre:Award Winning, genre:Comedy, genre:Fantasy, genre:Romance, genre:Sci-Fi, theme:Delinquents, theme:Martial Arts, theme:Mecha
- 隐藏目标：**Full Metal Panic!**（mal:71）· 公开评分 7.61 · 集数 24 · 人气排名 421/14182 · 标签 genre:Action, genre:Comedy, genre:Sci-Fi, theme:Mecha, theme:Military, theme:School · 制作公司 Gonzo
- 目标与画像 top-10 标签重叠 4 个（genre:Action, genre:Comedy, genre:Sci-Fi, theme:Mecha）；目标的制作公司未出现在可见收藏中
- 候选数 14179；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 256 | 0 | Saber Marionette J / Turn A Gundam / Pokemon Movie 01: Mewtw… / Dragon Ball Z Movie 12:… / Kidou Butouden G Gundam / Dragon Ball Z Movie 13:… / Dragon Ball Z Movie 08:… / Dragon Ball: Episode of… |
| A1 | 282 | 0 | Tengen Toppa Gurren Lag… / Code Geass: Hangyaku no… / Saber Marionette J / Tenkuu no Shiro Laputa / Sennen Joyuu / Turn A Gundam / Code Geass: Hangyaku no… / Mononoke Hime |
| B0_seed42 | 191 | 0 | Fullmetal Alchemist: Br… / Mononoke Hime / Death Note / Natsume Yuujinchou Roku / Tenkuu no Shiro Laputa / Code Geass: Hangyaku no… / One Piece / Tengen Toppa Gurren Lag… |
| B1_seed42 | 534 | 0 | Death Note / One Piece / Fullmetal Alchemist: Br… / Macross F Movie 2: Sayo… / Fairy Tail / Bleach / Dragon Ball Z / Fullmetal Alchemist |
| C0_seed42 | 69 | 0 | Naruto / One Piece / Fullmetal Alchemist / Naruto: Shippuuden / Fullmetal Alchemist: Br… / Death Note / Bleach / Code Geass: Hangyaku no… |

**观察**：A0 的 Top-8 平均公开评分 7.47、平均 hits 6.1；目标公开评分 7.61、hits 4。学习模型对目标的名次：B0_seed42 191，B1_seed42 534，C0_seed42 69。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏只有 3 部，每部占画像频率的 1/3，标签信号噪声大。

## 混合偏好（genre 熵高于验证集中位数）

测试集中符合条件的 query 数：1809。
### 案例 9：混合偏好（genre 熵高于验证集中位数） · query `test-10085-mal:437`

- 离线用户 10085 · 主组 · 可见收藏 36 部 · 规则 R0（无筛选）
- 可见收藏：Neon Genesis Evangelion；Neon Genesis Evangelion: The End of Eva…；Koukaku Kidoutai；Akira；Azumanga Daiou The Animation；Blood+；Mononoke Hime；Sen to Chihiro no Kamikakushi；Lodoss-tou Senki；Genshiken；…（共 36 部）
- 画像 top-10 标签：genre:Award Winning, genre:Drama, genre:Adventure, genre:Sci-Fi, genre:Action, genre:Comedy, genre:Romance, genre:Fantasy, theme:School, theme:Coming of Age
- 隐藏目标：**Perfect Blue**（mal:437）· 公开评分 8.54 · 集数 1 · 人气排名 288/14182 · 标签 genre:Avant Garde, genre:Drama, genre:Horror, theme:Adult Cast, theme:Psychological, theme:Showbiz · 制作公司 Madhouse
- 目标与画像 top-10 标签重叠 1 个（genre:Drama）；目标的制作公司出现在可见收藏中
- 候选数 14146；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 4856 | 0 | Urusei Yatsura Movie 1:… / Cardcaptor Sakura / Top wo Nerae! Gunbuster / Turn A Gundam / Urusei Yatsura / Blue Seed / Cardcaptor Sakura Movie… / Urusei Yatsura: Haru da… |
| A1 | 1365 | 0 | Cardcaptor Sakura / Top wo Nerae! Gunbuster / Turn A Gundam / Urusei Yatsura / Cardcaptor Sakura Movie… / Code Geass: Hangyaku no… / Fullmetal Alchemist / Toki wo Kakeru Shoujo |
| B0_seed42 | 638 | 0 | Fullmetal Alchemist: Br… / Fullmetal Alchemist / Toki wo Kakeru Shoujo / Code Geass: Hangyaku no… / Death Note / Evangelion: 1.0 You Are… / One Piece / Code Geass: Hangyaku no… |
| B1_seed42 | 30 | 0 | Toki wo Kakeru Shoujo / Fullmetal Alchemist: Br… / Kaze Tachinu / Tengen Toppa Gurren Lag… / Cardcaptor Sakura / Koukaku Kidoutai: Stand… / Kimi ni Todoke 2nd Seas… / Kimi ni Todoke |
| C0_seed42 | 82 | 0 | Code Geass: Hangyaku no… / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / Fullmetal Alchemist: Br… / Naruto: Shippuuden / Cowboy Bebop / One Piece |

**观察**：A0 的 Top-8 平均公开评分 7.54、平均 hits 7.0；目标公开评分 8.54、hits 1。学习模型对目标的名次：B0_seed42 638，B1_seed42 30，C0_seed42 82。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏的 genre 熵高于预先确定的阈值 3.33，偏好分散，top-10 标签难以代表单一口味。

### 案例 10：混合偏好（genre 熵高于验证集中位数） · query `test-101103-mal:4081`

- 离线用户 101103 · 主组 · 可见收藏 41 部 · 规则 R0（无筛选）
- 可见收藏：Cowboy Bebop；Cowboy Bebop: Tengoku no Tobira；Beck；Shin Kidou Senki Gundam Wing；Fruits Basket；Tenkuu no Escaflowne；Sen to Chihiro no Kamikakushi；Wolf's Rain；Samurai Champloo；FLCL；…（共 41 部）
- 画像 top-10 标签：genre:Action, genre:Drama, genre:Adventure, genre:Comedy, genre:Mystery, genre:Supernatural, genre:Fantasy, genre:Sci-Fi, genre:Award Winning, theme:Psychological
- 隐藏目标：**Natsume Yuujinchou**（mal:4081）· 公开评分 8.31 · 集数 13 · 人气排名 393/14182 · 标签 genre:Drama, genre:Slice of Life, genre:Supernatural, theme:Coming of Age, theme:Iyashikei, theme:School · 制作公司 Brain's Base
- 目标与画像 top-10 标签重叠 2 个（genre:Drama, genre:Supernatural）；目标的制作公司出现在可见收藏中
- 候选数 14141；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2165 | 0 | Serial Experiments Lain / One Piece Film: Red / Pokemon Movie 01: Mewtw… / RahXephon / Made in Abyss / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Evangelion: 3.0+1.0 Thr… |
| A1 | 629 | 0 | Made in Abyss / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Evangelion: 3.0+1.0 Thr… / Fullmetal Alchemist: Br… / Serial Experiments Lain / Mo Dao Zu Shi: Wanjie P… / Kenpuu Denki Berserk |
| B0_seed42 | 401 | 0 | Fullmetal Alchemist: Br… / Mononoke Hime / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Piece / Fate/stay night Movie: … / Made in Abyss Movie 3: … / Serial Experiments Lain |
| B1_seed42 | 167 | 0 | Fullmetal Alchemist: Br… / Mononoke Hime / Tengen Toppa Gurren Lag… / Kidou Senshi Gundam 00 / Code Geass: Hangyaku no… / Danshi Koukousei no Nic… / Danshi Koukousei no Nic… / One Piece |
| C0_seed42 | 307 | 0 | Code Geass: Hangyaku no… / Code Geass: Hangyaku no… / Fullmetal Alchemist / Death Note / Fullmetal Alchemist: Br… / Naruto: Shippuuden / One Piece / Tengen Toppa Gurren Lag… |

**观察**：A0 的 Top-8 平均公开评分 8.19、平均 hits 5.5；目标公开评分 8.31、hits 2。学习模型对目标的名次：B0_seed42 401，B1_seed42 167，C0_seed42 307。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏的 genre 熵高于预先确定的阈值 3.33，偏好分散，top-10 标签难以代表单一口味。

## 规则排除了隐藏目标（严格模式不插入）

测试集中符合条件的 query 数：5146。
### 案例 11：规则排除了隐藏目标（严格模式不插入） · query `test-101230-mal:196`

- 离线用户 101230 · 主组 · 可见收藏 105 部 · 规则 R2（公开评分 ≥ 7.0 且集数 ≤ 26）
- 可见收藏：Cowboy Bebop；Aa! Megami-sama! (TV)；Chobits；Chrno Crusade；D.N.Angel；Black Cat (TV)；Full Metal Panic? Fumoffu；Full Metal Panic! The Second Raid；Shuffle!；Mai-HiME；…（共 105 部）
- 画像 top-10 标签：genre:Comedy, genre:Romance, theme:School, genre:Action, genre:Fantasy, genre:Ecchi, genre:Supernatural, genre:Drama, genre:Adventure, genre:Sci-Fi
- 隐藏目标：**Onegai☆Twins**（mal:196）· 公开评分 6.82 · 集数 12 · 人气排名 2089/14182 · 标签 genre:Comedy, genre:Drama, genre:Sci-Fi, theme:Coming of Age, theme:School · 制作公司 Daume
- 目标与画像 top-10 标签重叠 4 个（genre:Comedy, genre:Drama, genre:Sci-Fi, theme:School）；目标的制作公司出现在可见收藏中
- 候选数 3598；目标不满足本规则（严格模式不会把它插回候选，该 query 不计入条件指标，端到端记为未找到）

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 不适用 | — | Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Urusei Yatsura Movie 1:… / Photon / Blue Seed / Gakusen Toshi Asterisk … / Urusei Yatsura 2: Beaut… |
| A1 | 不适用 | — | Gintama: The Final / Zero no Tsukaima: Futat… / Kaguya-sama wa Kokurase… / Urusei Yatsura 2: Beaut… / Zero no Tsukaima: Princ… / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Zero no Tsukaima |
| B0_seed42 | 不适用 | — | Code Geass: Hangyaku no… / Mononoke Hime / Fate/stay night Movie: … / Kaguya-sama wa Kokurase… / Tenkuu no Shiro Laputa / Zero no Tsukaima / Zero no Tsukaima: Princ… / Clannad: After Story |
| B1_seed42 | 不适用 | — | Code Geass: Hangyaku no… / Kono Subarashii Sekai n… / Clannad / Dungeon ni Deai wo Moto… / Higurashi no Naku Koro … / Bakuman. 2nd Season / Yuragi-sou no Yuuna-san / Zero no Tsukaima |
| C0_seed42 | 不适用 | — | Ouran Koukou Host Club / Clannad: After Story / Fruits Basket / Clannad / Junjou Romantica 2 / Lucky☆Star / Vampire Knight Guilty / Lovely★Complex |

**观察**：A0 的 Top-8 平均公开评分 7.23、平均 hits 6.9；目标公开评分 6.82、hits 4。目标被规则排除的原因：公开评分低于 7.0。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。目标在目录中较冷门（人气排名 2089），而各模型的主要信号与公开评分 / 人气相关，冷门目标普遍难以进入前 8。

### 案例 12：规则排除了隐藏目标（严格模式不插入） · query `test-101148-mal:182`

- 离线用户 101148 · 主组 · 可见收藏 12 部 · 规则 R1（关键词 = 可见收藏最常见 genre） · 关键词 `action`
- 可见收藏：Cowboy Bebop；Cowboy Bebop: Tengoku no Tobira；Trigun；Sen to Chihiro no Kamikakushi；Hellsing；Fate/stay night；Vampire Hunter D (2000)；Kyuuketsuhime Miyu (TV)；Vampire Hunter D；Higurashi no Naku Koro ni；…（共 12 部）
- 画像 top-10 标签：genre:Action, genre:Supernatural, theme:Gore, genre:Horror, genre:Sci-Fi, theme:Vampire, theme:Mythology, genre:Adventure, genre:Drama, genre:Fantasy
- 隐藏目标：**Tenkuu no Escaflowne**（mal:182）· 公开评分 7.67 · 集数 26 · 人气排名 1415/14182 · 标签 genre:Adventure, genre:Fantasy, genre:Romance, theme:Gore, theme:Isekai, theme:Mecha, theme:Mythology · 制作公司 Sunrise
- 目标与画像 top-10 标签重叠 4 个（genre:Adventure, genre:Fantasy, theme:Gore, theme:Mythology）；目标的制作公司出现在可见收藏中
- 候选数 3909；目标不满足本规则（严格模式不会把它插回候选，该 query 不计入条件指标，端到端记为未找到）

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 不适用 | — | Akira / X/1999 / Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Blood+ |
| A1 | 不适用 | — | Akira / Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Fullmetal Alchemist: Br… / Mo Dao Zu Shi: Wanjie P… / Kimetsu no Yaiba / Mo Dao Zu Shi / Mo Dao Zu Shi: Xian Yun… |
| B0_seed42 | 不适用 | — | Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / One Piece / Mononoke Hime / Fate/stay night Movie: … / Akira / Jigokuraku / Hellsing Ultimate |
| B1_seed42 | 不适用 | — | Fullmetal Alchemist: Br… / Hellsing Ultimate / Mononoke Hime / Code Geass: Hangyaku no… / Trigun: Badlands Rumble / Rurouni Kenshin: Meiji … / One Piece / Gungrave |
| C0_seed42 | 不适用 | — | Code Geass: Hangyaku no… / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Piece / Fullmetal Alchemist: Br… / Naruto: Shippuuden / Naruto / Tengen Toppa Gurren Lag… |

**观察**：A0 的 Top-8 平均公开评分 7.82、平均 hits 6.4；目标公开评分 7.67、hits 4。目标被规则排除的原因：genre 与标题都不含关键词“action”。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。

### 案例 13：规则排除了隐藏目标（严格模式不插入） · query `test-101534-mal:177`

- 离线用户 101534 · 主组 · 可见收藏 18 部 · 规则 R3（R1 + R2） · 关键词 `drama`
- 可见收藏：Aa! Megami-sama! (TV)；Rozen Maiden；Rozen Maiden: Träumend；Fruits Basket；Fullmetal Alchemist；Loveless；Mahou Sensei Negima!；Gravitation；Matantei Loki Ragnarok；Ouran Koukou Host Club；…（共 18 部）
- 画像 top-10 标签：genre:Drama, genre:Comedy, genre:Supernatural, genre:Romance, genre:Action, theme:School, genre:Adventure, genre:Fantasy, genre:Mystery, theme:Coming of Age
- 隐藏目标：**Tsubasa Chronicle**（mal:177）· 公开评分 7.52 · 集数 26 · 人气排名 1005/14182 · 标签 genre:Action, genre:Adventure, genre:Fantasy, genre:Romance, theme:Isekai, theme:Survival · 制作公司 Bee Train
- 目标与画像 top-10 标签重叠 4 个（genre:Action, genre:Adventure, genre:Fantasy, genre:Romance）；目标的制作公司出现在可见收藏中
- 候选数 886；目标不满足本规则（严格模式不会把它插回候选，该 query 不计入条件指标，端到端记为未找到）

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 不适用 | — | Urusei Yatsura Movie 1:… / Blue Seed / Sennen Joyuu / Little Busters! Refrain / Little Busters! EX / Kami no Tou / Sekai Saikou no Ansatsu… / Nakitai Watashi wa Neko… |
| A1 | 不适用 | — | Sennen Joyuu / Little Busters! Refrain / Mo Dao Zu Shi: Wanjie P… / Kara no Kyoukai Movie 5… / Mo Dao Zu Shi / Mo Dao Zu Shi: Xian Yun… / Houseki no Kuni / Wu Liuqi: Xuanwu Guo Pi… |
| B0_seed42 | 不适用 | — | Kaguya-sama wa Kokurase… / Kara no Kyoukai Movie 5… / Clannad: After Story / Bakuman. 2nd Season / Code Geass: Hangyaku no… / Made in Abyss Movie 3: … / Yuuki Yuuna wa Yuusha d… / Toki wo Kakeru Shoujo |
| B1_seed42 | 不适用 | — | Clannad: After Story / Little Busters! Refrain / Beck / Dennou Coil / Bakuman. 2nd Season / Toradora! / Chobits / Chihayafuru |
| C0_seed42 | 不适用 | — | Code Geass: Hangyaku no… / Code Geass: Hangyaku no… / Elfen Lied / Clannad: After Story / Vampire Knight Guilty / Junjou Romantica 2 / Chobits / Howl no Ugoku Shiro |

**观察**：A0 的 Top-8 平均公开评分 7.55、平均 hits 6.2；目标公开评分 7.52、hits 4。目标被规则排除的原因：genre 与标题都不含关键词“drama”。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。

## 严格空结果（候选数 = 0）

测试集中符合条件的 query 数：0。
本运行的测试集中**未出现**此类案例，因此不展示（不编造）。

## 候选不足 8 个

测试集中符合条件的 query 数：0。
本运行的测试集中**未出现**此类案例，因此不展示（不编造）。

