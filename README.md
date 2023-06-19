# sintetic-attention-validator

# Q1

## How is the salience model compared with the human eye track data

- the RMSE comparing the salience model with the demo samples is 330-ish

- the RMSE comparing the screen central point with the demo saples is 500-ish

- the RMSE comparing the salience model with the video samples is 450-ish

- the RMSE comparing the screen central point with the video saples is 590-ish

# Q2

## Wich group had the best results?

People that watched the video get more answers right
![ANOVA Right Answers](RightAns.png)

People that played the demo were less sure when answering the questions
![ANOVA dont know](DontKnow.png)

Anwers distribution
![graph](Answers.png)

# Q3

|Question             |sal Video|sal Demo|Forms video|Forms demo|
|---------------------|---------|--------|-----------|----------|
|Number of pudus      |4        |yes     |51%        |69%       |
|Bear action          |5        |yes     |74%        |91%       |
|Day or night         |-        |-       |100%       |100%      |
|sun on sky           |0        |no      |29%        |56%       |
|Trail material       |0        |no      |74%        |82%       |
|Dominant tree species|2        |no      |33%        |26%       |
|house's color        |5        |yes     |85%        |95%       |
|Birds                |4        |yes     |55%        |30%       |
|Pudus                |5        |yes     |92%        |95%       |
|Bear                 |5        |yes     |85%        |95%       |
