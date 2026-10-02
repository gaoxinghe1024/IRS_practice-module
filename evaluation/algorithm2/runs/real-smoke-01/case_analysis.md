# 失败案例分析（真实测试 query）

运行 `real-smoke-01`（级别 smoke）。案例全部取自 predictions.parquet 的测试 query，按 query_id 排序后在每类中选取，不挑选“好看”的例子。核心模型：A0、A1 与部署 seed 42 的 B0 / B1 / C0。

阅读说明：**观察**是从数据直接读出的事实；**待验证推测**只是可能原因，未经对照实验验证，特征重要性与案例相关性都不是因果解释。标签是代理标签（评分 ≥ 8），未评分的番剧是弱负例，“未命中”不代表用户不喜欢推荐结果。

| 类别 | 测试中出现次数 | 展示案例数 |
|---|---|---|
| A0 命中、学习模型（B1 与 C0）未命中 | 2 | 2 |
| 学习模型命中、A0 未命中 | 23 | 2 |
| 五个核心模型都未命中 | 124 | 2 |
| 稀疏收藏（可见历史 1–4 部） | 30 | 2 |
| 混合偏好（genre 熵高于验证集中位数） | 87 | 2 |
| 规则排除了隐藏目标（严格模式不插入） | 251 | 3 |
| 严格空结果（候选数 = 0） | 0 | 0 |
| 候选不足 8 个 | 0 | 0 |

## A0 命中、学习模型（B1 与 C0）未命中

测试集中符合条件的 query 数：2。
### 案例 1：A0 命中、学习模型（B1 与 C0）未命中 · query `test-20701-mal:543`

- 离线用户 20701 · 主组 · 可见收藏 17 部 · 规则 R0（无筛选）
- 可见收藏：Trigun；Naruto；Rurouni Kenshin: Meiji Kenkaku Romantan…；Appleseed (Movie)；Mononoke Hime；Tenjou Tenge；Sen to Chihiro no Kamikakushi；Elfen Lied；Hellsing；Blood: The Last Vampire；…（共 17 部）
- 画像 top-10 标签：genre:Action, genre:Supernatural, theme:Gore, genre:Adventure, genre:Horror, genre:Award Winning, genre:Fantasy, genre:Drama, genre:Romance, genre:Sci-Fi
- 隐藏目标：**Vampire Hunter D (2000)**（mal:543）· 公开评分 7.91 · 集数 1 · 人气排名 1238/14182 · 标签 genre:Action, genre:Drama, genre:Fantasy, genre:Horror, genre:Romance, genre:Sci-Fi, theme:Gore, theme:Vampire · 制作公司 Madhouse
- 目标与画像 top-10 标签重叠 7 个（genre:Action, genre:Drama, genre:Fantasy, genre:Horror, genre:Romance, genre:Sci-Fi, theme:Gore）；目标的制作公司出现在可见收藏中
- 候选数 14165；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 1 | 1 | Vampire Hunter D (2000) / Kenpuu Denki Berserk / Zetman / Sennen Joyuu / Berserk: Ougon Jidai-he… / Akira / X/1999 / Berserk: Ougon Jidai-he… |
| A1 | 2 | 1 | Kenpuu Denki Berserk / Vampire Hunter D (2000) / Sennen Joyuu / Berserk: Ougon Jidai-he… / Made in Abyss / Made in Abyss: Retsujit… / Akira / Made in Abyss Movie 3: … |
| B0_seed42 | 65 | 0 | Code Geass: Hangyaku no… / One Piece / Fate/stay night Movie: … / Shingeki no Kyojin Seas… / Fullmetal Alchemist: Br… / Fullmetal Alchemist / Cowboy Bebop / Violet Evergarden |
| B1_seed42 | 18 | 0 | Hunter x Hunter (2011) / Hellsing Ultimate / Bleach: Sennen Kessen-h… / Trigun: Badlands Rumble / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Made in Abyss / Claymore |
| C0_seed42 | 1007 | 0 | Naruto: Shippuuden / Toriko / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Punch Man / Nana / Bleach / Perfect Blue |

**观察**：A0 的 Top-8 平均公开评分 7.79、平均 hits 6.4；目标公开评分 7.91、hits 7。学习模型对目标的名次：B0_seed42 65，B1_seed42 18，C0_seed42 1007。

**待验证推测**：目标与画像标签高度重叠（hits 7，不低于 A0 推荐条目的平均值），A0 的 hits 项把它排在前面；学习模型更依赖公开评分（B1 中 rating_norm 的 gain 占比 61%），把评分或人气更高的条目排在它前面。

### 案例 2：A0 命中、学习模型（B1 与 C0）未命中 · query `test-72047-mal:232`

- 离线用户 72047 · 主组 · 可见收藏 12 部 · 规则 R0（无筛选）
- 可见收藏：.hack//Sign；Aa! Megami-sama!；Fullmetal Alchemist；.hack//Liminality；Fate/stay night；.hack//Gift；.hack//Roots；Hayate no Gotoku!；.hack//G.U. Returner；Blassreiter；…（共 12 部）
- 画像 top-10 标签：genre:Fantasy, genre:Comedy, genre:Action, genre:Adventure, genre:Drama, genre:Romance, theme:Coming of Age, theme:School, genre:Award Winning, genre:Mystery
- 隐藏目标：**Cardcaptor Sakura**（mal:232）· 公开评分 8.16 · 集数 70 · 人气排名 505/14182 · 标签 genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, genre:Romance, theme:Coming of Age, theme:Iyashikei, theme:Mahou Shoujo, theme:Martial Arts, theme:School · 制作公司 Madhouse
- 目标与画像 top-10 标签重叠 7 个（genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, genre:Romance, theme:Coming of Age, theme:School）；目标的制作公司出现在可见收藏中
- 候选数 14170；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2 | 1 | Sennen Joyuu / Cardcaptor Sakura / Urusei Yatsura Movie 1:… / Blue Seed / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Princess Tutu |
| A1 | 2 | 1 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Howl no Ugoku Shiro / Princess Tutu / Fullmetal Alchemist: Br… |
| B0_seed42 | 62 | 0 | Mononoke Hime / Code Geass: Hangyaku no… / One Piece / Spy x Family / Fullmetal Alchemist: Br… / Toki wo Kakeru Shoujo / Great Teacher Onizuka / Fate/stay night Movie: … |
| B1_seed42 | 170 | 0 | K-On! Movie / Violet Evergarden / Tsubasa Chronicle 2nd S… / Fullmetal Alchemist: Br… / Steins;Gate / Beck / Kono Subarashii Sekai n… / Code Geass: Hangyaku no… |
| C0_seed42 | 309 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana / Bleach |

**观察**：A0 的 Top-8 平均公开评分 7.90、平均 hits 6.5；目标公开评分 8.16、hits 7。学习模型对目标的名次：B0_seed42 62，B1_seed42 170，C0_seed42 309。

**待验证推测**：目标与画像标签高度重叠（hits 7，不低于 A0 推荐条目的平均值），A0 的 hits 项把它排在前面；学习模型更依赖公开评分（B1 中 rating_norm 的 gain 占比 61%），把评分或人气更高的条目排在它前面。

## 学习模型命中、A0 未命中

测试集中符合条件的 query 数：23。
### 案例 3：学习模型命中、A0 未命中 · query `test-1094645-mal:10271`

- 离线用户 1094645 · 主组 · 可见收藏 15 部 · 规则 R0（无筛选）
- 可见收藏：One Piece；Sakigake!! Cromartie Koukou；Gyakkyou Burai Kaiji: Ultimate Survivor；Fullmetal Alchemist: Brotherhood；Angel Beats!；Redline；Durarara!!；Mitsudomoe；Bakuman.；Steins;Gate；…（共 15 部）
- 画像 top-10 标签：theme:School, genre:Comedy, genre:Action, genre:Drama, genre:Adventure, genre:Fantasy, theme:Coming of Age, genre:Romance, genre:Sci-Fi, genre:Slice of Life
- 隐藏目标：**Gyakkyou Burai Kaiji: Hakairoku-hen**（mal:10271）· 公开评分 8.25 · 集数 26 · 人气排名 1181/14182 · 标签 genre:Suspense, theme:Psychological, theme:Survival · 制作公司 Madhouse
- 目标与画像 top-10 标签重叠 0 个；目标的制作公司出现在可见收藏中
- 候选数 14167；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 8683 | 0 | Urusei Yatsura Movie 1:… / Cardcaptor Sakura / Urusei Yatsura / Blue Seed / Sennen Joyuu / Urusei Yatsura: Haru da… / Bishoujo Senshi Sailor … / Kimi to Boku. 2 |
| A1 | 3707 | 0 | Cardcaptor Sakura / Sennen Joyuu / Urusei Yatsura / Urusei Yatsura Movie 1:… / Gintama: The Final / Bishoujo Senshi Sailor … / Kimi to Boku. 2 / Kaguya-sama wa Kokurase… |
| B0_seed42 | 1568 | 0 | Code Geass: Hangyaku no… / Mononoke Hime / Spy x Family / Fullmetal Alchemist / Toki wo Kakeru Shoujo / Tamayura: Sotsugyou Sha… / Great Teacher Onizuka / Fate/stay night Movie: … |
| B1_seed42 | 3 | 1 | Death Note / Hunter x Hunter (2011) / Gyakkyou Burai Kaiji: H… / Beck / Little Busters! Refrain / Sakura-sou no Pet na Ka… / Kimi to Boku. 2 / Lucky☆Star |
| C0_seed42 | 285 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**观察**：A0 的 Top-8 平均公开评分 7.58、平均 hits 6.8；目标公开评分 8.25、hits 0。学习模型对目标的名次：B0_seed42 1568，B1_seed42 3，C0_seed42 285。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。B1 / C0 使用了制作公司、集数、相似度等 A0 没有的信号（目标的制作公司出现在可见收藏中），可能因此排得更前；这只是相关观察，不是因果证明。

### 案例 4：学习模型命中、A0 未命中 · query `test-1122033-mal:7645`

- 离线用户 1122033 · 主组 · 可见收藏 102 部 · 规则 R0（无筛选）
- 可见收藏：Neon Genesis Evangelion；Neon Genesis Evangelion: The End of Eva…；Kenpuu Denki Berserk；Koukaku Kidoutai；Rurouni Kenshin: Meiji Kenkaku Romantan…；Rurouni Kenshin: Meiji Kenkaku Romantan；Mahou Shoujo Lyrical Nanoha；Mahou Shoujo Lyrical Nanoha A's；Samurai Champloo；Cardcaptor Sakura；…（共 102 部）
- 画像 top-10 标签：genre:Action, genre:Fantasy, genre:Comedy, genre:Adventure, theme:School, theme:Mahou Shoujo, genre:Drama, theme:Isekai, genre:Romance, theme:Samurai
- 隐藏目标：**Heartcatch Precure!**（mal:7645）· 公开评分 7.85 · 集数 49 · 人气排名 3658/14182 · 标签 genre:Action, genre:Comedy, genre:Fantasy, theme:Educational, theme:Mahou Shoujo, theme:Martial Arts, theme:School, theme:Superhero · 制作公司 Toei Animation
- 目标与画像 top-10 标签重叠 5 个（genre:Action, genre:Comedy, genre:Fantasy, theme:Mahou Shoujo, theme:School）；目标的制作公司出现在可见收藏中
- 候选数 14080；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 46 | 0 | Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Zero no Tsukaima: Princ… / Tsubasa: Tokyo Revelati… / Tsubasa: Shunraiki / Bishoujo Senshi Sailor … / Drifters |
| A1 | 52 | 0 | Tsubasa: Tokyo Revelati… / Tsubasa: Shunraiki / Gintama: The Final / Bishoujo Senshi Sailor … / Zero no Tsukaima: Futat… / Drifters / Tensei shitara Slime Da… / Kaguya-sama wa Kokurase… |
| B0_seed42 | 55 | 0 | One Piece / Mononoke Hime / Code Geass: Hangyaku no… / Mushoku Tensei: Isekai … / Spy x Family / Great Teacher Onizuka / Fate/stay night Movie: … / Violet Evergarden |
| B1_seed42 | 7 | 1 | Kobayashi-san Chi no Ma… / Precure All Stars Movie… / Violet Evergarden / Hirogaru Sky! Precure / One Piece / Berserk: Ougon Jidai-he… / Heartcatch Precure! / Code Geass: Hangyaku no… |
| C0_seed42 | 6335 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**观察**：A0 的 Top-8 平均公开评分 7.64、平均 hits 6.5；目标公开评分 7.85、hits 5。学习模型对目标的名次：B0_seed42 55，B1_seed42 7，C0_seed42 6335。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。目标在目录中较冷门（人气排名 3658），而各模型的主要信号与公开评分 / 人气相关，冷门目标普遍难以进入前 8。B1 / C0 使用了制作公司、集数、相似度等 A0 没有的信号（目标的制作公司出现在可见收藏中），可能因此排得更前；这只是相关观察，不是因果证明。

## 五个核心模型都未命中

测试集中符合条件的 query 数：124。
### 案例 5：五个核心模型都未命中 · query `test-10390-mal:16498`

- 离线用户 10390 · 主组 · 可见收藏 5 部 · 规则 R0（无筛选）
- 可见收藏：Fruits Basket；Ouran Koukou Host Club；Special A；Fullmetal Alchemist: Brotherhood；Kenka Banchou Otome: Girl Beats Boys
- 画像 top-10 标签：theme:School, genre:Romance, genre:Action, genre:Comedy, genre:Drama, theme:Crossdressing, theme:Delinquents, theme:Martial Arts, genre:Adventure, genre:Fantasy
- 隐藏目标：**Shingeki no Kyojin**（mal:16498）· 公开评分 8.54 · 集数 25 · 人气排名 1/14182 · 标签 genre:Action, genre:Award Winning, genre:Drama, genre:Suspense, theme:Gore, theme:Military, theme:Survival · 制作公司 Wit Studio
- 目标与画像 top-10 标签重叠 2 个（genre:Action, genre:Drama）；目标的制作公司未出现在可见收藏中
- 候选数 14177；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 1878 | 0 | Cardcaptor Sakura / Urusei Yatsura / Tenjou Tenge: The Past … / Dragon Ball Kai (2014) / Pokemon Movie 01: Mewtw… / Kimagure Orange☆Road / Da Wang Rao Ming / Zero no Tsukaima: Futat… |
| A1 | 421 | 0 | Cardcaptor Sakura / Urusei Yatsura / Wu Liuqi: Xuanwu Guo Pi… / Kaguya-sama wa Kokurase… / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Fruits Basket 1st Season / Cardcaptor Sakura Movie… |
| B0_seed42 | 42 | 0 | One Piece / Mononoke Hime / Great Teacher Onizuka / Code Geass: Hangyaku no… / Spy x Family / Fate/stay night Movie: … / Violet Evergarden / Shingeki no Kyojin Seas… |
| B1_seed42 | 124 | 0 | Fullmetal Alchemist / Boku no Hero Academia 6… / Fullmetal Alchemist: Br… / Fruits Basket 2nd Season / Seto no Hanayome / Soul Eater / Boku no Hero Academia 2… / Bounen no Xamdou |
| C0_seed42 | 17 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**观察**：A0 的 Top-8 平均公开评分 7.59、平均 hits 6.4；目标公开评分 8.54、hits 2。学习模型对目标的名次：B0_seed42 42，B1_seed42 124，C0_seed42 17。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。

### 案例 6：五个核心模型都未命中 · query `test-103940-mal:477`

- 离线用户 103940 · 主组 · 可见收藏 9 部 · 规则 R0（无筛选）
- 可见收藏：Haibane Renmei；Toradora!；One Outs；Ookami to Koushinryou II；Saki；Kobato.；Kimi ni Todoke；Tokyo Magnitude 8.0；Toaru Kagaku no Railgun
- 画像 top-10 标签：genre:Drama, theme:School, genre:Fantasy, genre:Romance, theme:Coming of Age, theme:Iyashikei, theme:Psychological, genre:Action, genre:Adventure, genre:Award Winning
- 隐藏目标：**Aria the Animation**（mal:477）· 公开评分 7.67 · 集数 13 · 人气排名 1370/14182 · 标签 genre:Fantasy, genre:Sci-Fi, genre:Slice of Life, theme:Iyashikei · 制作公司 Hal Film Maker
- 目标与画像 top-10 标签重叠 2 个（genre:Fantasy, theme:Iyashikei）；目标的制作公司未出现在可见收藏中
- 候选数 14173；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2270 | 0 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Fullmetal Alchemist / Evangelion: 1.0 You Are… / Kotonoha no Niwa / Turn A Gundam / Shoujo Kakumei Utena: A… |
| A1 | 1581 | 0 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Howl no Ugoku Shiro / Fullmetal Alchemist / Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / Evangelion: 1.0 You Are… |
| B0_seed42 | 1795 | 0 | Mononoke Hime / Code Geass: Hangyaku no… / One Piece / Spy x Family / Fullmetal Alchemist: Br… / Fullmetal Alchemist / Toki wo Kakeru Shoujo / Fate/stay night Movie: … |
| B1_seed42 | 781 | 0 | Sakura-sou no Pet na Ka… / Toaru Kagaku no Railgun… / Kimi ni Todoke 2nd Seas… / Bakuman. 2nd Season / Toki wo Kakeru Shoujo / Hajime no Ippo: New Cha… / Cardcaptor Sakura / Dennou Coil |
| C0_seed42 | 3215 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**观察**：A0 的 Top-8 平均公开评分 8.00、平均 hits 6.2；目标公开评分 7.67、hits 2。学习模型对目标的名次：B0_seed42 1795，B1_seed42 781，C0_seed42 3215。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。

## 稀疏收藏（可见历史 1–4 部）

测试集中符合条件的 query 数：30。
### 案例 7：稀疏收藏（可见历史 1–4 部） · query `test-104662-mal:916`

- 离线用户 104662 · 稀疏组 · 可见收藏 4 部 · 规则 R0（无筛选）
- 可见收藏：Samurai Champloo；Elfen Lied；FLCL；Bleach
- 画像 top-10 标签：genre:Action, genre:Adventure, genre:Comedy, genre:Avant Garde, genre:Drama, genre:Fantasy, genre:Horror, genre:Romance, genre:Sci-Fi, genre:Supernatural
- 隐藏目标：**Tokkou**（mal:916）· 公开评分 6.53 · 集数 13 · 人气排名 3011/14182 · 标签 genre:Action, genre:Comedy, genre:Horror, genre:Mystery, genre:Supernatural, theme:Gore · 制作公司 AIC Spirits, Group TAC
- 目标与画像 top-10 标签重叠 4 个（genre:Action, genre:Comedy, genre:Horror, genre:Supernatural）；目标的制作公司未出现在可见收藏中
- 候选数 14178；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 838 | 0 | Blue Seed / Silent Möbius / Vampire Hunter D (2000) / Urusei Yatsura 2: Beaut… / Urusei Yatsura / Urusei Yatsura Movie 5:… / Saber Marionette J / Macross 7 |
| A1 | 2341 | 0 | Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / Gintama: The Final / Vampire Hunter D (2000) / Urusei Yatsura 2: Beaut… / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Urusei Yatsura |
| B0_seed42 | 3382 | 0 | One Piece / Mononoke Hime / Fate/stay night Movie: … / Code Geass: Hangyaku no… / Fullmetal Alchemist: Br… / Violet Evergarden / Monogatari Series: Seco… / Natsume Yuujinchou Roku |
| B1_seed42 | 2192 | 0 | Bleach: Sennen Kessen-h… / Tengen Toppa Gurren Lag… / Bleach Movie 3: Fade to… / Bleach Movie 4: Jigoku-… / Naruto: Shippuuden / Tengen Toppa Gurren Lag… / Tengen Toppa Gurren Lag… / Black Clover: Mahou Tei… |
| C0_seed42 | 1223 | 0 | Naruto: Shippuuden / Toriko / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana / Perfect Blue |

**观察**：A0 的 Top-8 平均公开评分 7.41、平均 hits 6.2；目标公开评分 6.53、hits 4。学习模型对目标的名次：B0_seed42 3382，B1_seed42 2192，C0_seed42 1223。

**待验证推测**：目标的公开评分明显低于 A0 推荐条目的平均值；以公开评分为主要信号的排序可能把它压到后面。目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。目标在目录中较冷门（人气排名 3011），而各模型的主要信号与公开评分 / 人气相关，冷门目标普遍难以进入前 8。可见收藏只有 4 部，每部占画像频率的 1/4，标签信号噪声大。

### 案例 8：稀疏收藏（可见历史 1–4 部） · query `test-1076639-mal:21`

- 离线用户 1076639 · 稀疏组 · 可见收藏 2 部 · 规则 R0（无筛选）
- 可见收藏：Death Note；One Piece Film: Strong World Episode 0
- 画像 top-10 标签：genre:Adventure, genre:Fantasy, genre:Supernatural, genre:Suspense, theme:Psychological
- 隐藏目标：**One Piece**（mal:21）· 公开评分 8.69 · 集数 未知 · 人气排名 20/14182 · 标签 genre:Action, genre:Adventure, genre:Fantasy · 制作公司 Toei Animation
- 目标与画像 top-10 标签重叠 2 个（genre:Adventure, genre:Fantasy）；目标的制作公司出现在可见收藏中
- 候选数 14180；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 60 | 0 | Kino no Tabi: The Beaut… / Kenpuu Denki Berserk / Yakusoku no Neverland / Banana Fish / Re:Zero kara Hajimeru I… / Mononoke / Fumetsu no Anata e / Re:Zero kara Hajimeru I… |
| A1 | 19 | 0 | Fullmetal Alchemist: Br… / Steins;Gate / Bleach: Sennen Kessen-h… / Kenpuu Denki Berserk / Hunter x Hunter (2011) / Yakusoku no Neverland / Banana Fish / Re:Zero kara Hajimeru I… |
| B0_seed42 | 1 | 1 | One Piece / Mononoke Hime / Fate/stay night Movie: … / Mushishi Zoku Shou / Summertime Render / Ping Pong the Animation / Shingeki no Kyojin Seas… / The First Slam Dunk |
| B1_seed42 | 2 | 1 | One Piece 3D2Y: Ace no … / One Piece / Hunter x Hunter (2011) / Death Note: Rewrite / One Piece Film: Gold / One Piece: Episode of E… / One Piece Film: Strong … / Death Billiards |
| C0_seed42 | 36 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Punch Man / Nana / Bleach |

**观察**：A0 的 Top-8 平均公开评分 8.33、平均 hits 3.1；目标公开评分 8.69、hits 2。学习模型对目标的名次：B0_seed42 1，B1_seed42 2，C0_seed42 36。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏只有 2 部，每部占画像频率的 1/2，标签信号噪声大。

## 混合偏好（genre 熵高于验证集中位数）

测试集中符合条件的 query 数：87。
### 案例 9：混合偏好（genre 熵高于验证集中位数） · query `test-1065515-mal:9515`

- 离线用户 1065515 · 主组 · 可见收藏 89 部 · 规则 R0（无筛选）
- 可见收藏：Cowboy Bebop；Cowboy Bebop: Tengoku no Tobira；Trigun；Hachimitsu to Clover；Sunabouzu；Full Metal Panic!；Full Metal Panic? Fumoffu；Full Metal Panic! The Second Raid；Green Green；Gunslinger Girl；…（共 89 部）
- 画像 top-10 标签：genre:Comedy, theme:School, genre:Action, genre:Romance, genre:Drama, genre:Sci-Fi, genre:Supernatural, genre:Ecchi, genre:Adventure, genre:Fantasy
- 隐藏目标：**Highschool of the Dead: Drifters of the Dead**（mal:9515）· 公开评分 6.62 · 集数 1 · 人气排名 741/14182 · 标签 genre:Comedy, genre:Ecchi, genre:Supernatural, theme:Crossdressing, theme:Survival · 制作公司 Madhouse
- 目标与画像 top-10 标签重叠 3 个（genre:Comedy, genre:Ecchi, genre:Supernatural）；目标的制作公司出现在可见收藏中
- 候选数 14093；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 3227 | 0 | Urusei Yatsura / Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Urusei Yatsura Movie 1:… / Photon / Blue Seed / Gakusen Toshi Asterisk … |
| A1 | 4218 | 0 | Urusei Yatsura / Cardcaptor Sakura / Fullmetal Alchemist: Br… / Gintama: The Final / Zero no Tsukaima: Futat… / Kaguya-sama wa Kokurase… / Urusei Yatsura 2: Beaut… / Zero no Tsukaima: Princ… |
| B0_seed42 | 4167 | 0 | One Piece / Fate/stay night Movie: … / Mushoku Tensei: Isekai … / Spy x Family / Fullmetal Alchemist: Br… / Violet Evergarden / Monogatari Series: Seco… / Natsume Yuujinchou Roku |
| B1_seed42 | 4305 | 0 | Kidou Senshi Gundam: Th… / Hellsing / Bungou Stray Dogs 4th S… / Little Busters! Refrain / Fullmetal Alchemist: Br… / Josee to Tora to Sakana… / Zoku Sayonara Zetsubou … / Zan Sayonara Zetsubou S… |
| C0_seed42 | 1076 | 0 | Naruto: Shippuuden / Toriko / Fullmetal Alchemist / One Punch Man / Nana / Bleach / Perfect Blue / Mirai Nikki (TV) |

**观察**：A0 的 Top-8 平均公开评分 7.22、平均 hits 7.0；目标公开评分 6.62、hits 3。学习模型对目标的名次：B0_seed42 4167，B1_seed42 4305，C0_seed42 1076。

**待验证推测**：目标的公开评分明显低于 A0 推荐条目的平均值；以公开评分为主要信号的排序可能把它压到后面。目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏的 genre 熵高于预先确定的阈值 3.31，偏好分散，top-10 标签难以代表单一口味。

### 案例 10：混合偏好（genre 熵高于验证集中位数） · query `test-1101919-mal:14719`

- 离线用户 1101919 · 主组 · 可见收藏 116 部 · 规则 R0（无筛选）
- 可见收藏：Cowboy Bebop；Trigun；Neon Genesis Evangelion；Neon Genesis Evangelion: The End of Eva…；Kenpuu Denki Berserk；Koukaku Kidoutai；Rurouni Kenshin: Meiji Kenkaku Romantan；Akira；Azumanga Daiou The Animation；Full Metal Panic!；…（共 116 部）
- 画像 top-10 标签：genre:Comedy, genre:Action, theme:School, genre:Sci-Fi, genre:Adventure, genre:Drama, genre:Award Winning, genre:Fantasy, theme:Gore, theme:Military
- 隐藏目标：**JoJo no Kimyou na Bouken (TV)**（mal:14719）· 公开评分 7.89 · 集数 26 · 人气排名 59/14182 · 标签 genre:Action, genre:Adventure, genre:Supernatural, theme:Gore, theme:Historical · 制作公司 David Production
- 目标与画像 top-10 标签重叠 3 个（genre:Action, genre:Adventure, theme:Gore）；目标的制作公司未出现在可见收藏中
- 候选数 14066；目标满足本规则

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 1035 | 0 | Top wo Nerae! Gunbuster / Urusei Yatsura / Code Geass: Hangyaku no… / Fullmetal Alchemist / One Piece Film: Red / Berserk: Ougon Jidai-he… / Kidou Senshi Gundam SEED / Pokemon Movie 01: Mewtw… |
| A1 | 589 | 0 | Code Geass: Hangyaku no… / Fullmetal Alchemist: Br… / Code Geass: Hangyaku no… / Top wo Nerae! Gunbuster / Urusei Yatsura / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Fullmetal Alchemist |
| B0_seed42 | 168 | 0 | Code Geass: Hangyaku no… / One Piece / Shingeki no Kyojin Seas… / Spy x Family / Fullmetal Alchemist: Br… / Shingeki no Kyojin: The… / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… |
| B1_seed42 | 823 | 0 | Made in Abyss: Retsujit… / Hibike! Euphonium 2 / Berserk: Ougon Jidai-he… / Made in Abyss Movie 3: … / K-On!! / Kobayashi-san Chi no Ma… / Kidou Senshi Gundam: Su… / Code Geass: Hangyaku no… |
| C0_seed42 | 1155 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / Nana / Bleach |

**观察**：A0 的 Top-8 平均公开评分 7.94、平均 hits 6.2；目标公开评分 7.89、hits 3。学习模型对目标的名次：B0_seed42 168，B1_seed42 823，C0_seed42 1155。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏的 genre 熵高于预先确定的阈值 3.31，偏好分散，top-10 标签难以代表单一口味。

## 规则排除了隐藏目标（严格模式不插入）

测试集中符合条件的 query 数：251。
### 案例 11：规则排除了隐藏目标（严格模式不插入） · query `test-1054861-mal:3588`

- 离线用户 1054861 · 主组 · 可见收藏 10 部 · 规则 R2（公开评分 ≥ 7.0 且集数 ≤ 26）
- 可见收藏：Naruto；One Piece；Dragon Ball；Hellsing；Pokemon；Dragon Ball Z；Death Note；Naruto: Shippuuden；Fullmetal Alchemist: Brotherhood；Hunter x Hunter (2011)
- 画像 top-10 标签：genre:Action, genre:Adventure, genre:Fantasy, theme:Martial Arts, genre:Comedy, theme:Military, genre:Supernatural, theme:Coming of Age, theme:Gore, genre:Drama
- 隐藏目标：**Soul Eater**（mal:3588）· 公开评分 7.84 · 集数 51 · 人气排名 62/14182 · 标签 genre:Action, genre:Comedy, genre:Fantasy, theme:Martial Arts, theme:Samurai, theme:School · 制作公司 Bones
- 目标与画像 top-10 标签重叠 4 个（genre:Action, genre:Comedy, genre:Fantasy, theme:Martial Arts）；目标的制作公司出现在可见收藏中
- 候选数 3658；目标不满足本规则（严格模式不会把它插回候选，该 query 不计入条件指标，端到端记为未找到）

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 不适用 | — | Da Wang Rao Ming / Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Koutetsujou no Kabaneri… / Ima, Soko ni Iru Boku |
| A1 | 不适用 | — | Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… / Da Wang Rao Ming / Wu Liuqi: Xuanwu Guo Pi… / Berserk: Ougon Jidai-he… / Fumetsu no Anata e |
| B0_seed42 | 不适用 | — | Mononoke Hime / Fate/stay night Movie: … / Code Geass: Hangyaku no… / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… / Spy x Family / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… |
| B1_seed42 | 不适用 | — | Bleach: Sennen Kessen-h… / Hellsing Ultimate / Dragon Ball Super: Broly / One Piece Film: Strong … / One Piece: Episode of N… / One Piece Film: Z / Saint Seiya: Meiou Hade… / Dragon Ball Z Movie 12:… |
| C0_seed42 | 不适用 | — | Elfen Lied / Code Geass: Hangyaku no… / One Punch Man / Perfect Blue / Code Geass: Hangyaku no… / Mirai Nikki (TV) / Koukaku Kidoutai: Stand… / Kaze no Tani no Nausicaä |

**观察**：A0 的 Top-8 平均公开评分 7.88、平均 hits 6.1；目标公开评分 7.84、hits 4。目标被规则排除的原因：集数 51 > 26。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。

### 案例 12：规则排除了隐藏目标（严格模式不插入） · query `test-1083015-mal:4181`

- 离线用户 1083015 · 稀疏组 · 可见收藏 3 部 · 规则 R1（关键词 = 可见收藏最常见 genre） · 关键词 `comedy`
- 可见收藏：Kiss x Sis (TV)；B-gata H-kei；Mahou Shoujo Madoka★Magica
- 画像 top-10 标签：theme:School, genre:Comedy, genre:Ecchi, genre:Romance, theme:Coming of Age, genre:Award Winning, genre:Drama, genre:Suspense, theme:Mahou Shoujo, theme:Psychological
- 隐藏目标：**Clannad: After Story**（mal:4181）· 公开评分 8.93 · 集数 24 · 人气排名 114/14182 · 标签 genre:Drama, genre:Romance, genre:Supernatural · 制作公司 Kyoto Animation
- 目标与画像 top-10 标签重叠 2 个（genre:Drama, genre:Romance）；目标的制作公司未出现在可见收藏中
- 候选数 4997；目标不满足本规则（严格模式不会把它插回候选，该 query 不计入条件指标，端到端记为未找到）

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 不适用 | — | Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Cardcaptor Sakura / Princess Tutu / Chobits / Nakitai Watashi wa Neko… / Mahou no Stage Fancy La… / Bakuman. 2nd Season |
| A1 | 不适用 | — | Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Cardcaptor Sakura / Princess Tutu / Kaguya-sama wa Kokurase… / Bakuman. 2nd Season / NHK ni Youkoso! / Bakuman. |
| B0_seed42 | 不适用 | — | Spy x Family / Great Teacher Onizuka / Bakuman. 2nd Season / NHK ni Youkoso! / Kaguya-sama wa Kokurase… / Princess Tutu / Kaguya-sama wa Kokurase… / Kodomo no Omocha (TV) |
| B1_seed42 | 不适用 | — | Yahari Ore no Seishun L… / Yahari Ore no Seishun L… / Kaguya-sama wa Kokurase… / Yahari Ore no Seishun L… / Hidamari Sketch x Honey… / Kaguya-sama wa Kokurase… / Hinamatsuri (TV) / Kono Bijutsu-bu ni wa M… |
| C0_seed42 | 不适用 | — | Toriko / One Punch Man / Soul Eater / Great Teacher Onizuka / Suzumiya Haruhi no Yuuu… / Kill la Kill / Samurai Champloo / Katekyo Hitman Reborn! |

**观察**：A0 的 Top-8 平均公开评分 7.87、平均 hits 5.9；目标公开评分 8.93、hits 2。目标被规则排除的原因：genre 与标题都不含关键词“comedy”。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。可见收藏只有 3 部，每部占画像频率的 1/3，标签信号噪声大。

### 案例 13：规则排除了隐藏目标（严格模式不插入） · query `test-107950-mal:223`

- 离线用户 107950 · 主组 · 可见收藏 5 部 · 规则 R3（R1 + R2） · 关键词 `adventure`
- 可见收藏：Dragon Ball Z；Dragon Ball Movie 4: Saikyou e no Michi；Dragon Ball Z Special 2: Zetsubou e no …；Dragon Ball Z Special 1: Tatta Hitori n…；Dragon Ball Kai
- 画像 top-10 标签：genre:Adventure, genre:Fantasy, genre:Comedy, theme:Martial Arts, genre:Action, theme:Space, genre:Sci-Fi, genre:Drama, theme:Coming of Age, theme:Gore
- 隐藏目标：**Dragon Ball**（mal:223）· 公开评分 7.96 · 集数 153 · 人气排名 149/14182 · 标签 genre:Action, genre:Adventure, genre:Comedy, genre:Fantasy, theme:Martial Arts · 制作公司 Toei Animation
- 目标与画像 top-10 标签重叠 5 个（genre:Action, genre:Adventure, genre:Comedy, genre:Fantasy, theme:Martial Arts）；目标的制作公司出现在可见收藏中
- 候选数 781；目标不满足本规则（严格模式不会把它插回候选，该 query 不计入条件指标，端到端记为未找到）

| 模型 | 目标名次 | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 不适用 | — | Dragon Ball: Episode of… / Urusei Yatsura Movie 1:… / Kidou Senshi Gundam III… / Ima, Soko ni Iru Boku / Dragon Ball Z Movie 12:… / Da Wang Rao Ming / Urusei Yatsura Movie 5:… / Saber Marionette J |
| A1 | 不适用 | — | Made in Abyss / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Kenpuu Denki Berserk / Wu Liuqi: Xuanwu Guo Pi… / Vinland Saga Season 2 / Katanagatari / Jigokuraku |
| B0_seed42 | 不适用 | — | Mononoke Hime / Jigokuraku / Made in Abyss / Made in Abyss Movie 3: … / Made in Abyss: Retsujit… / Katanagatari / Wu Liuqi: Xuanwu Guo Pi… / Bleach: Sennen Kessen-h… |
| B1_seed42 | 不适用 | — | Dragon Ball Super: Broly / Dragon Ball Z Movie 12:… / Dragon Ball Z Movie 14:… / Dragon Ball Z Movie 08:… / Dragon Ball Z Movie 13:… / Dragon Ball Z Movie 09:… / One Piece Movie 08: Epi… / Dragon Ball Z Movie 05:… |
| C0_seed42 | 不适用 | — | Kaze no Tani no Nausicaä / Mononoke Hime / Howl no Ugoku Shiro / Samurai Champloo / Tenkuu no Shiro Laputa / Sen to Chihiro no Kamik… / Bleach: Sennen Kessen-h… / Tsubasa Chronicle |

**观察**：A0 的 Top-8 平均公开评分 7.44、平均 hits 6.2；目标公开评分 7.96、hits 5。目标被规则排除的原因：集数 153 > 26。

**待验证推测**：目标与画像 top-10 标签的重叠少于 A0 推荐条目，A0 的 hits 项不利于它。

## 严格空结果（候选数 = 0）

测试集中符合条件的 query 数：0。
本运行的测试集中**未出现**此类案例，因此不展示（不编造）。

## 候选不足 8 个

测试集中符合条件的 query 数：0。
本运行的测试集中**未出现**此类案例，因此不展示（不编造）。

