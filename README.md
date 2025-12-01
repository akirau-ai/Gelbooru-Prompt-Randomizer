# Gelbooru Prompt Randomizer

Generate Foreverできるように、チェックを入れると自動でプロンプトに追記できるように改変しました。  
ついでにプロンプトに投げるときに指定ワードを除外可能に。  
include指定はカンマ区切りで、_のスペース変換はしていないので入力時は cute_girl みたいに _ が必要です。  
コードはAIに丸投げして生成しました。

履歴
・　v1.0.0：or検索対応。{A|B|C}でAかBかCのどれかを含む。
・（）があるタグに対応。それに伴いor指定を（）から｛｝に変更。{a|b|c}。

I modified the script so that enabling “Generate Forever” automatically adds the required text to the prompt.  
I also added an option to exclude specific words when sending a prompt.  
The include field accepts comma-separated values, and it does not convert underscores to spaces, so please enter them as-is (e.g., cute_girl).  
All code was generated entirely by AI.

---

An extension for the [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), which randomly takes tags from a random post on the site [Gelbooru](https://gelbooru.com/), while you can set tags that should be in the post and that should not be.

<img src="./GPR.gif">
