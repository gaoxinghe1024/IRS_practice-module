# Failure-case analysis (real test queries)

Run `real-smoke-01` (level smoke). Every case is drawn from the test queries in predictions.parquet, ordered by query_id and then picked within each category; no 'nice looking' example was hand-picked. Core models: A0, A1 and B0 / B1 / C0 at the deploy seed 42.

How to read this: **Observation** is a fact read directly from the data; **Hypothesis to test** is only a possible cause, not verified by a controlled experiment — neither feature importance nor case correlation is a causal explanation. Labels are proxy labels (rating ≥ 8), unrated titles are weak negatives, and a 'miss' does not mean the user would dislike the recommendation.

| Category | Occurrences in the test set | Cases shown |
|---|---|---|
| A0 hits, learned models (B1 and C0) miss | 2 | 2 |
| A learned model hits, A0 misses | 23 | 2 |
| All five core models miss | 124 | 2 |
| Sparse collections (1–4 visible titles) | 30 | 2 |
| Mixed preferences (genre entropy above the validation median) | 87 | 2 |
| Rule excludes the hidden target (strict mode does not insert it) | 251 | 3 |
| Strict empty result (candidate count = 0) | 0 | 0 |
| Fewer than 8 candidates | 0 | 0 |

## A0 hits, learned models (B1 and C0) miss

Queries in the test set that fall into this category: 2.
### Case 1: A0 hits, learned models (B1 and C0) miss · query `test-20701-mal:543`

- Offline user 20701 · main group · 17 visible titles · rule R0 (no filter)
- Visible collection: Trigun; Naruto; Rurouni Kenshin: Meiji Kenkaku Romantan…; Appleseed (Movie); Mononoke Hime; Tenjou Tenge; Sen to Chihiro no Kamikakushi; Elfen Lied; Hellsing; Blood: The Last Vampire; … (17 titles)
- Profile top-10 tags: genre:Action, genre:Supernatural, theme:Gore, genre:Adventure, genre:Horror, genre:Award Winning, genre:Fantasy, genre:Drama, genre:Romance, genre:Sci-Fi
- Hidden target: **Vampire Hunter D (2000)** (mal:543) · public score 7.91 · episodes 1 · popularity rank 1238/14182 · tags genre:Action, genre:Drama, genre:Fantasy, genre:Horror, genre:Romance, genre:Sci-Fi, theme:Gore, theme:Vampire · studio Madhouse
- Overlap between the target and the profile top-10 tags: 7 (genre:Action, genre:Drama, genre:Fantasy, genre:Horror, genre:Romance, genre:Sci-Fi, theme:Gore); the target's studio appears in the visible collection
- Candidates: 14165; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 1 | 1 | Vampire Hunter D (2000) / Kenpuu Denki Berserk / Zetman / Sennen Joyuu / Berserk: Ougon Jidai-he… / Akira / X/1999 / Berserk: Ougon Jidai-he… |
| A1 | 2 | 1 | Kenpuu Denki Berserk / Vampire Hunter D (2000) / Sennen Joyuu / Berserk: Ougon Jidai-he… / Made in Abyss / Made in Abyss: Retsujit… / Akira / Made in Abyss Movie 3: … |
| B0_seed42 | 65 | 0 | Code Geass: Hangyaku no… / One Piece / Fate/stay night Movie: … / Shingeki no Kyojin Seas… / Fullmetal Alchemist: Br… / Fullmetal Alchemist / Cowboy Bebop / Violet Evergarden |
| B1_seed42 | 18 | 0 | Hunter x Hunter (2011) / Hellsing Ultimate / Bleach: Sennen Kessen-h… / Trigun: Badlands Rumble / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Made in Abyss / Claymore |
| C0_seed42 | 1007 | 0 | Naruto: Shippuuden / Toriko / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Punch Man / Nana / Bleach / Perfect Blue |

**Observation**: A0's Top-8 mean public score 7.79, mean hits 6.4; the target has public score 7.91 and hits 7. Learned models' rank for the target: B0_seed42 65, B1_seed42 18, C0_seed42 1007.

**Hypothesis (to verify)**: the target overlaps the profile tags heavily (hits 7, not below the average of the A0 recommendations), so A0's hits term ranks it high; the learned models lean more on public score (61% of B1's gain is rating_norm) and put titles with a higher score or popularity above it.

### Case 2: A0 hits, learned models (B1 and C0) miss · query `test-72047-mal:232`

- Offline user 72047 · main group · 12 visible titles · rule R0 (no filter)
- Visible collection: .hack//Sign; Aa! Megami-sama!; Fullmetal Alchemist; .hack//Liminality; Fate/stay night; .hack//Gift; .hack//Roots; Hayate no Gotoku!; .hack//G.U. Returner; Blassreiter; … (12 titles)
- Profile top-10 tags: genre:Fantasy, genre:Comedy, genre:Action, genre:Adventure, genre:Drama, genre:Romance, theme:Coming of Age, theme:School, genre:Award Winning, genre:Mystery
- Hidden target: **Cardcaptor Sakura** (mal:232) · public score 8.16 · episodes 70 · popularity rank 505/14182 · tags genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, genre:Romance, theme:Coming of Age, theme:Iyashikei, theme:Mahou Shoujo, theme:Martial Arts, theme:School · studio Madhouse
- Overlap between the target and the profile top-10 tags: 7 (genre:Adventure, genre:Comedy, genre:Drama, genre:Fantasy, genre:Romance, theme:Coming of Age, theme:School); the target's studio appears in the visible collection
- Candidates: 14170; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2 | 1 | Sennen Joyuu / Cardcaptor Sakura / Urusei Yatsura Movie 1:… / Blue Seed / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Princess Tutu |
| A1 | 2 | 1 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Howl no Ugoku Shiro / Princess Tutu / Fullmetal Alchemist: Br… |
| B0_seed42 | 62 | 0 | Mononoke Hime / Code Geass: Hangyaku no… / One Piece / Spy x Family / Fullmetal Alchemist: Br… / Toki wo Kakeru Shoujo / Great Teacher Onizuka / Fate/stay night Movie: … |
| B1_seed42 | 170 | 0 | K-On! Movie / Violet Evergarden / Tsubasa Chronicle 2nd S… / Fullmetal Alchemist: Br… / Steins;Gate / Beck / Kono Subarashii Sekai n… / Code Geass: Hangyaku no… |
| C0_seed42 | 309 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana / Bleach |

**Observation**: A0's Top-8 mean public score 7.90, mean hits 6.5; the target has public score 8.16 and hits 7. Learned models' rank for the target: B0_seed42 62, B1_seed42 170, C0_seed42 309.

**Hypothesis (to verify)**: the target overlaps the profile tags heavily (hits 7, not below the average of the A0 recommendations), so A0's hits term ranks it high; the learned models lean more on public score (61% of B1's gain is rating_norm) and put titles with a higher score or popularity above it.

## A learned model hits, A0 misses

Queries in the test set that fall into this category: 23.
### Case 3: A learned model hits, A0 misses · query `test-1094645-mal:10271`

- Offline user 1094645 · main group · 15 visible titles · rule R0 (no filter)
- Visible collection: One Piece; Sakigake!! Cromartie Koukou; Gyakkyou Burai Kaiji: Ultimate Survivor; Fullmetal Alchemist: Brotherhood; Angel Beats!; Redline; Durarara!!; Mitsudomoe; Bakuman.; Steins;Gate; … (15 titles)
- Profile top-10 tags: theme:School, genre:Comedy, genre:Action, genre:Drama, genre:Adventure, genre:Fantasy, theme:Coming of Age, genre:Romance, genre:Sci-Fi, genre:Slice of Life
- Hidden target: **Gyakkyou Burai Kaiji: Hakairoku-hen** (mal:10271) · public score 8.25 · episodes 26 · popularity rank 1181/14182 · tags genre:Suspense, theme:Psychological, theme:Survival · studio Madhouse
- Overlap between the target and the profile top-10 tags: 0; the target's studio appears in the visible collection
- Candidates: 14167; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 8683 | 0 | Urusei Yatsura Movie 1:… / Cardcaptor Sakura / Urusei Yatsura / Blue Seed / Sennen Joyuu / Urusei Yatsura: Haru da… / Bishoujo Senshi Sailor … / Kimi to Boku. 2 |
| A1 | 3707 | 0 | Cardcaptor Sakura / Sennen Joyuu / Urusei Yatsura / Urusei Yatsura Movie 1:… / Gintama: The Final / Bishoujo Senshi Sailor … / Kimi to Boku. 2 / Kaguya-sama wa Kokurase… |
| B0_seed42 | 1568 | 0 | Code Geass: Hangyaku no… / Mononoke Hime / Spy x Family / Fullmetal Alchemist / Toki wo Kakeru Shoujo / Tamayura: Sotsugyou Sha… / Great Teacher Onizuka / Fate/stay night Movie: … |
| B1_seed42 | 3 | 1 | Death Note / Hunter x Hunter (2011) / Gyakkyou Burai Kaiji: H… / Beck / Little Busters! Refrain / Sakura-sou no Pet na Ka… / Kimi to Boku. 2 / Lucky☆Star |
| C0_seed42 | 285 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**Observation**: A0's Top-8 mean public score 7.58, mean hits 6.8; the target has public score 8.25 and hits 0. Learned models' rank for the target: B0_seed42 1568, B1_seed42 3, C0_seed42 285.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.B1 / C0 use signals A0 does not have (studio, episode count, similarity) and the target's studio appears in the visible collection, which may rank it higher; this is a correlational observation, not causal proof.

### Case 4: A learned model hits, A0 misses · query `test-1122033-mal:7645`

- Offline user 1122033 · main group · 102 visible titles · rule R0 (no filter)
- Visible collection: Neon Genesis Evangelion; Neon Genesis Evangelion: The End of Eva…; Kenpuu Denki Berserk; Koukaku Kidoutai; Rurouni Kenshin: Meiji Kenkaku Romantan…; Rurouni Kenshin: Meiji Kenkaku Romantan; Mahou Shoujo Lyrical Nanoha; Mahou Shoujo Lyrical Nanoha A's; Samurai Champloo; Cardcaptor Sakura; … (102 titles)
- Profile top-10 tags: genre:Action, genre:Fantasy, genre:Comedy, genre:Adventure, theme:School, theme:Mahou Shoujo, genre:Drama, theme:Isekai, genre:Romance, theme:Samurai
- Hidden target: **Heartcatch Precure!** (mal:7645) · public score 7.85 · episodes 49 · popularity rank 3658/14182 · tags genre:Action, genre:Comedy, genre:Fantasy, theme:Educational, theme:Mahou Shoujo, theme:Martial Arts, theme:School, theme:Superhero · studio Toei Animation
- Overlap between the target and the profile top-10 tags: 5 (genre:Action, genre:Comedy, genre:Fantasy, theme:Mahou Shoujo, theme:School); the target's studio appears in the visible collection
- Candidates: 14080; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 46 | 0 | Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Zero no Tsukaima: Princ… / Tsubasa: Tokyo Revelati… / Tsubasa: Shunraiki / Bishoujo Senshi Sailor … / Drifters |
| A1 | 52 | 0 | Tsubasa: Tokyo Revelati… / Tsubasa: Shunraiki / Gintama: The Final / Bishoujo Senshi Sailor … / Zero no Tsukaima: Futat… / Drifters / Tensei shitara Slime Da… / Kaguya-sama wa Kokurase… |
| B0_seed42 | 55 | 0 | One Piece / Mononoke Hime / Code Geass: Hangyaku no… / Mushoku Tensei: Isekai … / Spy x Family / Great Teacher Onizuka / Fate/stay night Movie: … / Violet Evergarden |
| B1_seed42 | 7 | 1 | Kobayashi-san Chi no Ma… / Precure All Stars Movie… / Violet Evergarden / Hirogaru Sky! Precure / One Piece / Berserk: Ougon Jidai-he… / Heartcatch Precure! / Code Geass: Hangyaku no… |
| C0_seed42 | 6335 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**Observation**: A0's Top-8 mean public score 7.64, mean hits 6.5; the target has public score 7.85 and hits 5. Learned models' rank for the target: B0_seed42 55, B1_seed42 7, C0_seed42 6335.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the target is relatively obscure in the catalogue (popularity rank 3658), the main signals of every model relate to public score / popularity, and obscure targets rarely reach the top 8.B1 / C0 use signals A0 does not have (studio, episode count, similarity) and the target's studio appears in the visible collection, which may rank it higher; this is a correlational observation, not causal proof.

## All five core models miss

Queries in the test set that fall into this category: 124.
### Case 5: All five core models miss · query `test-10390-mal:16498`

- Offline user 10390 · main group · 5 visible titles · rule R0 (no filter)
- Visible collection: Fruits Basket; Ouran Koukou Host Club; Special A; Fullmetal Alchemist: Brotherhood; Kenka Banchou Otome: Girl Beats Boys
- Profile top-10 tags: theme:School, genre:Romance, genre:Action, genre:Comedy, genre:Drama, theme:Crossdressing, theme:Delinquents, theme:Martial Arts, genre:Adventure, genre:Fantasy
- Hidden target: **Shingeki no Kyojin** (mal:16498) · public score 8.54 · episodes 25 · popularity rank 1/14182 · tags genre:Action, genre:Award Winning, genre:Drama, genre:Suspense, theme:Gore, theme:Military, theme:Survival · studio Wit Studio
- Overlap between the target and the profile top-10 tags: 2 (genre:Action, genre:Drama); the target's studio does not appear in the visible collection
- Candidates: 14177; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 1878 | 0 | Cardcaptor Sakura / Urusei Yatsura / Tenjou Tenge: The Past … / Dragon Ball Kai (2014) / Pokemon Movie 01: Mewtw… / Kimagure Orange☆Road / Da Wang Rao Ming / Zero no Tsukaima: Futat… |
| A1 | 421 | 0 | Cardcaptor Sakura / Urusei Yatsura / Wu Liuqi: Xuanwu Guo Pi… / Kaguya-sama wa Kokurase… / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Fruits Basket 1st Season / Cardcaptor Sakura Movie… |
| B0_seed42 | 42 | 0 | One Piece / Mononoke Hime / Great Teacher Onizuka / Code Geass: Hangyaku no… / Spy x Family / Fate/stay night Movie: … / Violet Evergarden / Shingeki no Kyojin Seas… |
| B1_seed42 | 124 | 0 | Fullmetal Alchemist / Boku no Hero Academia 6… / Fullmetal Alchemist: Br… / Fruits Basket 2nd Season / Seto no Hanayome / Soul Eater / Boku no Hero Academia 2… / Bounen no Xamdou |
| C0_seed42 | 17 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**Observation**: A0's Top-8 mean public score 7.59, mean hits 6.4; the target has public score 8.54 and hits 2. Learned models' rank for the target: B0_seed42 42, B1_seed42 124, C0_seed42 17.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.

### Case 6: All five core models miss · query `test-103940-mal:477`

- Offline user 103940 · main group · 9 visible titles · rule R0 (no filter)
- Visible collection: Haibane Renmei; Toradora!; One Outs; Ookami to Koushinryou II; Saki; Kobato.; Kimi ni Todoke; Tokyo Magnitude 8.0; Toaru Kagaku no Railgun
- Profile top-10 tags: genre:Drama, theme:School, genre:Fantasy, genre:Romance, theme:Coming of Age, theme:Iyashikei, theme:Psychological, genre:Action, genre:Adventure, genre:Award Winning
- Hidden target: **Aria the Animation** (mal:477) · public score 7.67 · episodes 13 · popularity rank 1370/14182 · tags genre:Fantasy, genre:Sci-Fi, genre:Slice of Life, theme:Iyashikei · studio Hal Film Maker
- Overlap between the target and the profile top-10 tags: 2 (genre:Fantasy, theme:Iyashikei); the target's studio does not appear in the visible collection
- Candidates: 14173; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 2270 | 0 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Fullmetal Alchemist / Evangelion: 1.0 You Are… / Kotonoha no Niwa / Turn A Gundam / Shoujo Kakumei Utena: A… |
| A1 | 1581 | 0 | Sennen Joyuu / Cardcaptor Sakura / Majo no Takkyuubin / Howl no Ugoku Shiro / Fullmetal Alchemist / Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / Evangelion: 1.0 You Are… |
| B0_seed42 | 1795 | 0 | Mononoke Hime / Code Geass: Hangyaku no… / One Piece / Spy x Family / Fullmetal Alchemist: Br… / Fullmetal Alchemist / Toki wo Kakeru Shoujo / Fate/stay night Movie: … |
| B1_seed42 | 781 | 0 | Sakura-sou no Pet na Ka… / Toaru Kagaku no Railgun… / Kimi ni Todoke 2nd Seas… / Bakuman. 2nd Season / Toki wo Kakeru Shoujo / Hajime no Ippo: New Cha… / Cardcaptor Sakura / Dennou Coil |
| C0_seed42 | 3215 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana |

**Observation**: A0's Top-8 mean public score 8.00, mean hits 6.2; the target has public score 7.67 and hits 2. Learned models' rank for the target: B0_seed42 1795, B1_seed42 781, C0_seed42 3215.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.

## Sparse collections (1–4 visible titles)

Queries in the test set that fall into this category: 30.
### Case 7: Sparse collections (1–4 visible titles) · query `test-104662-mal:916`

- Offline user 104662 · sparse group · 4 visible titles · rule R0 (no filter)
- Visible collection: Samurai Champloo; Elfen Lied; FLCL; Bleach
- Profile top-10 tags: genre:Action, genre:Adventure, genre:Comedy, genre:Avant Garde, genre:Drama, genre:Fantasy, genre:Horror, genre:Romance, genre:Sci-Fi, genre:Supernatural
- Hidden target: **Tokkou** (mal:916) · public score 6.53 · episodes 13 · popularity rank 3011/14182 · tags genre:Action, genre:Comedy, genre:Horror, genre:Mystery, genre:Supernatural, theme:Gore · studio AIC Spirits, Group TAC
- Overlap between the target and the profile top-10 tags: 4 (genre:Action, genre:Comedy, genre:Horror, genre:Supernatural); the target's studio does not appear in the visible collection
- Candidates: 14178; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 838 | 0 | Blue Seed / Silent Möbius / Vampire Hunter D (2000) / Urusei Yatsura 2: Beaut… / Urusei Yatsura / Urusei Yatsura Movie 5:… / Saber Marionette J / Macross 7 |
| A1 | 2341 | 0 | Fullmetal Alchemist: Br… / Kenpuu Denki Berserk / Gintama: The Final / Vampire Hunter D (2000) / Urusei Yatsura 2: Beaut… / Tsubasa: Tokyo Revelati… / Sennen Joyuu / Urusei Yatsura |
| B0_seed42 | 3382 | 0 | One Piece / Mononoke Hime / Fate/stay night Movie: … / Code Geass: Hangyaku no… / Fullmetal Alchemist: Br… / Violet Evergarden / Monogatari Series: Seco… / Natsume Yuujinchou Roku |
| B1_seed42 | 2192 | 0 | Bleach: Sennen Kessen-h… / Tengen Toppa Gurren Lag… / Bleach Movie 3: Fade to… / Bleach Movie 4: Jigoku-… / Naruto: Shippuuden / Tengen Toppa Gurren Lag… / Tengen Toppa Gurren Lag… / Black Clover: Mahou Tei… |
| C0_seed42 | 1223 | 0 | Naruto: Shippuuden / Toriko / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / One Punch Man / Nana / Perfect Blue |

**Observation**: A0's Top-8 mean public score 7.41, mean hits 6.2; the target has public score 6.53 and hits 4. Learned models' rank for the target: B0_seed42 3382, B1_seed42 2192, C0_seed42 1223.

**Hypothesis (to verify)**: the target's public score is clearly below the average of the A0 recommendations, so an ordering driven mainly by public score may push it down.the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the target is relatively obscure in the catalogue (popularity rank 3011), the main signals of every model relate to public score / popularity, and obscure targets rarely reach the top 8.the visible collection holds only 4 titles, each contributing 1/4 of the profile frequency, so the tag signal is noisy.

### Case 8: Sparse collections (1–4 visible titles) · query `test-1076639-mal:21`

- Offline user 1076639 · sparse group · 2 visible titles · rule R0 (no filter)
- Visible collection: Death Note; One Piece Film: Strong World Episode 0
- Profile top-10 tags: genre:Adventure, genre:Fantasy, genre:Supernatural, genre:Suspense, theme:Psychological
- Hidden target: **One Piece** (mal:21) · public score 8.69 · episodes unknown · popularity rank 20/14182 · tags genre:Action, genre:Adventure, genre:Fantasy · studio Toei Animation
- Overlap between the target and the profile top-10 tags: 2 (genre:Adventure, genre:Fantasy); the target's studio appears in the visible collection
- Candidates: 14180; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 60 | 0 | Kino no Tabi: The Beaut… / Kenpuu Denki Berserk / Yakusoku no Neverland / Banana Fish / Re:Zero kara Hajimeru I… / Mononoke / Fumetsu no Anata e / Re:Zero kara Hajimeru I… |
| A1 | 19 | 0 | Fullmetal Alchemist: Br… / Steins;Gate / Bleach: Sennen Kessen-h… / Kenpuu Denki Berserk / Hunter x Hunter (2011) / Yakusoku no Neverland / Banana Fish / Re:Zero kara Hajimeru I… |
| B0_seed42 | 1 | 1 | One Piece / Mononoke Hime / Fate/stay night Movie: … / Mushishi Zoku Shou / Summertime Render / Ping Pong the Animation / Shingeki no Kyojin Seas… / The First Slam Dunk |
| B1_seed42 | 2 | 1 | One Piece 3D2Y: Ace no … / One Piece / Hunter x Hunter (2011) / Death Note: Rewrite / One Piece Film: Gold / One Piece: Episode of E… / One Piece Film: Strong … / Death Billiards |
| C0_seed42 | 36 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / One Punch Man / Nana / Bleach |

**Observation**: A0's Top-8 mean public score 8.33, mean hits 3.1; the target has public score 8.69 and hits 2. Learned models' rank for the target: B0_seed42 1, B1_seed42 2, C0_seed42 36.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection holds only 2 titles, each contributing 1/2 of the profile frequency, so the tag signal is noisy.

## Mixed preferences (genre entropy above the validation median)

Queries in the test set that fall into this category: 87.
### Case 9: Mixed preferences (genre entropy above the validation median) · query `test-1065515-mal:9515`

- Offline user 1065515 · main group · 89 visible titles · rule R0 (no filter)
- Visible collection: Cowboy Bebop; Cowboy Bebop: Tengoku no Tobira; Trigun; Hachimitsu to Clover; Sunabouzu; Full Metal Panic!; Full Metal Panic? Fumoffu; Full Metal Panic! The Second Raid; Green Green; Gunslinger Girl; … (89 titles)
- Profile top-10 tags: genre:Comedy, theme:School, genre:Action, genre:Romance, genre:Drama, genre:Sci-Fi, genre:Supernatural, genre:Ecchi, genre:Adventure, genre:Fantasy
- Hidden target: **Highschool of the Dead: Drifters of the Dead** (mal:9515) · public score 6.62 · episodes 1 · popularity rank 741/14182 · tags genre:Comedy, genre:Ecchi, genre:Supernatural, theme:Crossdressing, theme:Survival · studio Madhouse
- Overlap between the target and the profile top-10 tags: 3 (genre:Comedy, genre:Ecchi, genre:Supernatural); the target's studio appears in the visible collection
- Candidates: 14093; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 3227 | 0 | Urusei Yatsura / Zero no Tsukaima: Futat… / Zero no Tsukaima: Princ… / Zero no Tsukaima / Urusei Yatsura Movie 1:… / Photon / Blue Seed / Gakusen Toshi Asterisk … |
| A1 | 4218 | 0 | Urusei Yatsura / Cardcaptor Sakura / Fullmetal Alchemist: Br… / Gintama: The Final / Zero no Tsukaima: Futat… / Kaguya-sama wa Kokurase… / Urusei Yatsura 2: Beaut… / Zero no Tsukaima: Princ… |
| B0_seed42 | 4167 | 0 | One Piece / Fate/stay night Movie: … / Mushoku Tensei: Isekai … / Spy x Family / Fullmetal Alchemist: Br… / Violet Evergarden / Monogatari Series: Seco… / Natsume Yuujinchou Roku |
| B1_seed42 | 4305 | 0 | Kidou Senshi Gundam: Th… / Hellsing / Bungou Stray Dogs 4th S… / Little Busters! Refrain / Fullmetal Alchemist: Br… / Josee to Tora to Sakana… / Zoku Sayonara Zetsubou … / Zan Sayonara Zetsubou S… |
| C0_seed42 | 1076 | 0 | Naruto: Shippuuden / Toriko / Fullmetal Alchemist / One Punch Man / Nana / Bleach / Perfect Blue / Mirai Nikki (TV) |

**Observation**: A0's Top-8 mean public score 7.22, mean hits 7.0; the target has public score 6.62 and hits 3. Learned models' rank for the target: B0_seed42 4167, B1_seed42 4305, C0_seed42 1076.

**Hypothesis (to verify)**: the target's public score is clearly below the average of the A0 recommendations, so an ordering driven mainly by public score may push it down.the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection's genre entropy is above the pre-determined threshold 3.31, the preferences are dispersed, and the top-10 tags hardly represent a single taste.

### Case 10: Mixed preferences (genre entropy above the validation median) · query `test-1101919-mal:14719`

- Offline user 1101919 · main group · 116 visible titles · rule R0 (no filter)
- Visible collection: Cowboy Bebop; Trigun; Neon Genesis Evangelion; Neon Genesis Evangelion: The End of Eva…; Kenpuu Denki Berserk; Koukaku Kidoutai; Rurouni Kenshin: Meiji Kenkaku Romantan; Akira; Azumanga Daiou The Animation; Full Metal Panic!; … (116 titles)
- Profile top-10 tags: genre:Comedy, genre:Action, theme:School, genre:Sci-Fi, genre:Adventure, genre:Drama, genre:Award Winning, genre:Fantasy, theme:Gore, theme:Military
- Hidden target: **JoJo no Kimyou na Bouken (TV)** (mal:14719) · public score 7.89 · episodes 26 · popularity rank 59/14182 · tags genre:Action, genre:Adventure, genre:Supernatural, theme:Gore, theme:Historical · studio David Production
- Overlap between the target and the profile top-10 tags: 3 (genre:Action, genre:Adventure, theme:Gore); the target's studio does not appear in the visible collection
- Candidates: 14066; the target satisfies this rule

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | 1035 | 0 | Top wo Nerae! Gunbuster / Urusei Yatsura / Code Geass: Hangyaku no… / Fullmetal Alchemist / One Piece Film: Red / Berserk: Ougon Jidai-he… / Kidou Senshi Gundam SEED / Pokemon Movie 01: Mewtw… |
| A1 | 589 | 0 | Code Geass: Hangyaku no… / Fullmetal Alchemist: Br… / Code Geass: Hangyaku no… / Top wo Nerae! Gunbuster / Urusei Yatsura / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Fullmetal Alchemist |
| B0_seed42 | 168 | 0 | Code Geass: Hangyaku no… / One Piece / Shingeki no Kyojin Seas… / Spy x Family / Fullmetal Alchemist: Br… / Shingeki no Kyojin: The… / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… |
| B1_seed42 | 823 | 0 | Made in Abyss: Retsujit… / Hibike! Euphonium 2 / Berserk: Ougon Jidai-he… / Made in Abyss Movie 3: … / K-On!! / Kobayashi-san Chi no Ma… / Kidou Senshi Gundam: Su… / Code Geass: Hangyaku no… |
| C0_seed42 | 1155 | 0 | Naruto: Shippuuden / Toriko / Elfen Lied / Fullmetal Alchemist / Code Geass: Hangyaku no… / Death Note / Nana / Bleach |

**Observation**: A0's Top-8 mean public score 7.94, mean hits 6.2; the target has public score 7.89 and hits 3. Learned models' rank for the target: B0_seed42 168, B1_seed42 823, C0_seed42 1155.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection's genre entropy is above the pre-determined threshold 3.31, the preferences are dispersed, and the top-10 tags hardly represent a single taste.

## Rule excludes the hidden target (strict mode does not insert it)

Queries in the test set that fall into this category: 251.
### Case 11: Rule excludes the hidden target (strict mode does not insert it) · query `test-1054861-mal:3588`

- Offline user 1054861 · main group · 10 visible titles · rule R2 (public score ≥ 7.0 and episodes ≤ 26)
- Visible collection: Naruto; One Piece; Dragon Ball; Hellsing; Pokemon; Dragon Ball Z; Death Note; Naruto: Shippuuden; Fullmetal Alchemist: Brotherhood; Hunter x Hunter (2011)
- Profile top-10 tags: genre:Action, genre:Adventure, genre:Fantasy, theme:Martial Arts, genre:Comedy, theme:Military, genre:Supernatural, theme:Coming of Age, theme:Gore, genre:Drama
- Hidden target: **Soul Eater** (mal:3588) · public score 7.84 · episodes 51 · popularity rank 62/14182 · tags genre:Action, genre:Comedy, genre:Fantasy, theme:Martial Arts, theme:Samurai, theme:School · studio Bones
- Overlap between the target and the profile top-10 tags: 4 (genre:Action, genre:Comedy, genre:Fantasy, theme:Martial Arts); the target's studio appears in the visible collection
- Candidates: 3658; the target does not satisfy this rule (strict mode never inserts it back into the candidates, so this query is excluded from the conditional metrics and counted as not found end to end)

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | n/a | — | Da Wang Rao Ming / Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Berserk: Ougon Jidai-he… / Koutetsujou no Kabaneri… / Ima, Soko ni Iru Boku |
| A1 | n/a | — | Kenpuu Denki Berserk / Berserk: Ougon Jidai-he… / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… / Da Wang Rao Ming / Wu Liuqi: Xuanwu Guo Pi… / Berserk: Ougon Jidai-he… / Fumetsu no Anata e |
| B0_seed42 | n/a | — | Mononoke Hime / Fate/stay night Movie: … / Code Geass: Hangyaku no… / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… / Spy x Family / Shingeki no Kyojin Seas… / Shingeki no Kyojin: The… |
| B1_seed42 | n/a | — | Bleach: Sennen Kessen-h… / Hellsing Ultimate / Dragon Ball Super: Broly / One Piece Film: Strong … / One Piece: Episode of N… / One Piece Film: Z / Saint Seiya: Meiou Hade… / Dragon Ball Z Movie 12:… |
| C0_seed42 | n/a | — | Elfen Lied / Code Geass: Hangyaku no… / One Punch Man / Perfect Blue / Code Geass: Hangyaku no… / Mirai Nikki (TV) / Koukaku Kidoutai: Stand… / Kaze no Tani no Nausicaä |

**Observation**: A0's Top-8 mean public score 7.88, mean hits 6.1; the target has public score 7.84 and hits 4. Why the rule excludes the target: episodes 51 > 26.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.

### Case 12: Rule excludes the hidden target (strict mode does not insert it) · query `test-1083015-mal:4181`

- Offline user 1083015 · sparse group · 3 visible titles · rule R1 (keyword = most common genre in the visible collection) · keyword `comedy``
- Visible collection: Kiss x Sis (TV); B-gata H-kei; Mahou Shoujo Madoka★Magica
- Profile top-10 tags: theme:School, genre:Comedy, genre:Ecchi, genre:Romance, theme:Coming of Age, genre:Award Winning, genre:Drama, genre:Suspense, theme:Mahou Shoujo, theme:Psychological
- Hidden target: **Clannad: After Story** (mal:4181) · public score 8.93 · episodes 24 · popularity rank 114/14182 · tags genre:Drama, genre:Romance, genre:Supernatural · studio Kyoto Animation
- Overlap between the target and the profile top-10 tags: 2 (genre:Drama, genre:Romance); the target's studio does not appear in the visible collection
- Candidates: 4997; the target does not satisfy this rule (strict mode never inserts it back into the candidates, so this query is excluded from the conditional metrics and counted as not found end to end)

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | n/a | — | Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Cardcaptor Sakura / Princess Tutu / Chobits / Nakitai Watashi wa Neko… / Mahou no Stage Fancy La… / Bakuman. 2nd Season |
| A1 | n/a | — | Cardcaptor Sakura Movie… / Shoujo Kakumei Utena / Cardcaptor Sakura / Princess Tutu / Kaguya-sama wa Kokurase… / Bakuman. 2nd Season / NHK ni Youkoso! / Bakuman. |
| B0_seed42 | n/a | — | Spy x Family / Great Teacher Onizuka / Bakuman. 2nd Season / NHK ni Youkoso! / Kaguya-sama wa Kokurase… / Princess Tutu / Kaguya-sama wa Kokurase… / Kodomo no Omocha (TV) |
| B1_seed42 | n/a | — | Yahari Ore no Seishun L… / Yahari Ore no Seishun L… / Kaguya-sama wa Kokurase… / Yahari Ore no Seishun L… / Hidamari Sketch x Honey… / Kaguya-sama wa Kokurase… / Hinamatsuri (TV) / Kono Bijutsu-bu ni wa M… |
| C0_seed42 | n/a | — | Toriko / One Punch Man / Soul Eater / Great Teacher Onizuka / Suzumiya Haruhi no Yuuu… / Kill la Kill / Samurai Champloo / Katekyo Hitman Reborn! |

**Observation**: A0's Top-8 mean public score 7.87, mean hits 5.9; the target has public score 8.93 and hits 2. Why the rule excludes the target: neither the genres nor the title contain the keyword `comedy`.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.the visible collection holds only 3 titles, each contributing 1/3 of the profile frequency, so the tag signal is noisy.

### Case 13: Rule excludes the hidden target (strict mode does not insert it) · query `test-107950-mal:223`

- Offline user 107950 · main group · 5 visible titles · rule R3 (R1 + R2) · keyword `adventure``
- Visible collection: Dragon Ball Z; Dragon Ball Movie 4: Saikyou e no Michi; Dragon Ball Z Special 2: Zetsubou e no …; Dragon Ball Z Special 1: Tatta Hitori n…; Dragon Ball Kai
- Profile top-10 tags: genre:Adventure, genre:Fantasy, genre:Comedy, theme:Martial Arts, genre:Action, theme:Space, genre:Sci-Fi, genre:Drama, theme:Coming of Age, theme:Gore
- Hidden target: **Dragon Ball** (mal:223) · public score 7.96 · episodes 153 · popularity rank 149/14182 · tags genre:Action, genre:Adventure, genre:Comedy, genre:Fantasy, theme:Martial Arts · studio Toei Animation
- Overlap between the target and the profile top-10 tags: 5 (genre:Action, genre:Adventure, genre:Comedy, genre:Fantasy, theme:Martial Arts); the target's studio appears in the visible collection
- Candidates: 781; the target does not satisfy this rule (strict mode never inserts it back into the candidates, so this query is excluded from the conditional metrics and counted as not found end to end)

| Model | Target rank | Hit@8 | Top-8 |
|---|---|---|---|
| A0 | n/a | — | Dragon Ball: Episode of… / Urusei Yatsura Movie 1:… / Kidou Senshi Gundam III… / Ima, Soko ni Iru Boku / Dragon Ball Z Movie 12:… / Da Wang Rao Ming / Urusei Yatsura Movie 5:… / Saber Marionette J |
| A1 | n/a | — | Made in Abyss / Made in Abyss: Retsujit… / Made in Abyss Movie 3: … / Kenpuu Denki Berserk / Wu Liuqi: Xuanwu Guo Pi… / Vinland Saga Season 2 / Katanagatari / Jigokuraku |
| B0_seed42 | n/a | — | Mononoke Hime / Jigokuraku / Made in Abyss / Made in Abyss Movie 3: … / Made in Abyss: Retsujit… / Katanagatari / Wu Liuqi: Xuanwu Guo Pi… / Bleach: Sennen Kessen-h… |
| B1_seed42 | n/a | — | Dragon Ball Super: Broly / Dragon Ball Z Movie 12:… / Dragon Ball Z Movie 14:… / Dragon Ball Z Movie 08:… / Dragon Ball Z Movie 13:… / Dragon Ball Z Movie 09:… / One Piece Movie 08: Epi… / Dragon Ball Z Movie 05:… |
| C0_seed42 | n/a | — | Kaze no Tani no Nausicaä / Mononoke Hime / Howl no Ugoku Shiro / Samurai Champloo / Tenkuu no Shiro Laputa / Sen to Chihiro no Kamik… / Bleach: Sennen Kessen-h… / Tsubasa Chronicle |

**Observation**: A0's Top-8 mean public score 7.44, mean hits 6.2; the target has public score 7.96 and hits 5. Why the rule excludes the target: episodes 153 > 26.

**Hypothesis (to verify)**: the target overlaps the profile top-10 tags less than the A0 recommendations do, so A0's hits term works against it.

## Strict empty result (candidate count = 0)

Queries in the test set that fall into this category: 0.
This category **did not occur** in this run's test set, so it is not shown (never invented).

## Fewer than 8 candidates

Queries in the test set that fall into this category: 0.
This category **did not occur** in this run's test set, so it is not shown (never invented).

