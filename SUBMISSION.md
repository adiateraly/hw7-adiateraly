# HW7 submission — Blackjack

GitHub: adiateraly · Course: CSS-4007 Artificial Intelligence, Fall 2026

## AI-use declaration

I used **Claude** (an AI assistant made by Anthropic) in this homework.

## Setup and reproduction

| Tool | Version |
| --- | --- |
| Python | 3.13.9 |
| gymnasium | 1.4.0 |
| numpy | 2.5.5 |
| matplotlib | 3.11.2 |
| System | Windows 11 |

The versions are also saved in `outputs/fixed-policy/versions.json` and `outputs/q-learning/versions.json`.

Install (from the repository folder):

```bash
python -m venv .venv
.venv\Scripts\activate            
pip install -r requirements.txt
```

Commands (run them in this order):

| What | Command | Output folder |
| --- | --- | --- |
| Tests | `python -m pytest -q` | – |
| Part C: replay of the paper games | `python -m blackjack_rl.paper_replay` | `outputs/paper-replay/` |
| Part D: fixed policy, 3 methods, 3 seeds | `python -m blackjack_rl.fixed_policy` | `outputs/fixed-policy/` |
| Part D: summary of method differences | `python -m blackjack_rl.compare_methods` | `outputs/fixed-policy/` |
| Part D: Q-learning and evaluation | `python -m blackjack_rl.q_experiment` | `outputs/q-learning/` |

All random numbers use fixed seeds, so the results can be repeated. The environment is `Blackjack-v1` with `natural=False, sab=False`. Settings: discount 1.0, learning rate 0.1, tables start at zero.

## Blackjack in my own words

- **State (what the learner sees):** three numbers: my card sum, the dealer's showing card (1–10, ace = 1) and a flag "usable ace" (0 or 1). The dealer's hidden card is not in the state.
- **Actions:** 0 = stick (stop), 1 = hit (take one more card).
- **Rewards:** +1 if I win, −1 if I lose, 0 for a draw. A hit that does not bust gives 0.
- **Termination:** the game ends when I stick (then the dealer plays and the result is decided) or when I bust (sum over 21). After that there is no next state. In the learning code this is the marker `END` and it has value 0.

**Why the usable-ace flag must be in the state.** Two hands can have the same sum but a different future. With a usable ace, a hit cannot bust me, because the ace can change from 11 to 1. Without it, a hit can bust me. In the supplied histories, C is "17 with usable ace" and D is "17 without". In game 3, I hit at C, draw a ten, and the ace becomes 1: the sum stays 17, I do not bust. At D a ten would give 27 and I would bust. So the best action can be different for the same sum. If the flag is not in the state, the learner mixes two different situations into one average.

## Part A — Count states and actions

Paper work (original, unchanged): [`paper/page1-partA.png`](paper/page1-partA.png), [`paper/page2-partA-partB.png`](paper/page2-partA-partB.png). The answers below repeat my paper results. The comparison table at the end of this part is new.

### 1. Blackjack

- **Observation entries:** sums 12–21 give 10 values, dealer showing 1–10 gives 10 values, usable ace gives 2 values. So 10 · 10 · 2 = **200** entries. This count is **exact for this restricted table**.
- **Action labels:** 2 (stick, hit). **Q-table entries:** 200 · 2 = **400**.
- **What the count does not include:** sums 4–11 and the terminal marker `END`. For example, the first hand can be 2+3 = 5. So 200 is not the size of the whole state space of the environment. The full game state also has the exact cards of the player and the dealer, including the dealer's hidden card.
- **Same observation, different situation:** player {10, 8} and player {9, 9}. Both have sum 18, no usable ace. The dealer shows 6, and the hidden card can be a 10 or a 5. The learner sees the same observation in both cases.
- **Why the learner must not see the hidden card:** a real player cannot see it. A policy that uses it could not be used in a real game. It would also hide the main difficulty, which is to decide with missing information.

### 2. Durak (two players, 36 cards, no transfers)

- **Possible hands:** C(36, 6) = 36! / (6! · 30!) = **1,947,792**. Order is ignored because a hand is a set of cards: the same six cards in a different order are the same hand. A hand alone is **not** a game state. We also need the opponent's hand, the trump card, the stock, the discard pile, the table, the phase and whose turn it is.
- **Candidate opponent hands:** I know my 6 cards and 1 exposed trump card. So 36 − 6 − 1 = 29 cards are unseen, and C(29, 6) = **475,020** opponent hands are possible. This is only a part of the hidden information. The 23 cards that are left in the stock are also in an unknown order, and the order changes future draws. So the full number of hidden states is larger. I do not give one number for all Durak states.
- **Opening attack:** the attacker plays one card. With six distinct cards there are **6** legal opening actions.
- **Concrete defense:** hearts are trumps, the attack is 10♣, my hand is 6♣, 9♣, A♣, 7♥, Q♥, 8♠.
  - Legal: A♣ (a higher club), 7♥ and Q♥ (trumps beat a non-trump), and **pick up**.
  - Not legal: 6♣ and 9♣ (clubs but lower than 10), 8♠ (wrong suit, not a trump).
  - Count: **4** legal actions.
- **Full simulator state:** both hands, the attack/defense pairs on the table, the trump suit and trump card, the stock (with its order), the discard pile, the phase (attack, defend, add cards) and whose turn it is.
- **What the player observes:** own hand, the table, the trump card, the size of the stock and of the opponent's hand. The same visible table can hide different opponent hands and a different stock order, so the future can be different.
- **Action labels:** I use "play one card" (36 labels), "pick up" and "end attack" = **38** labels. Only 4 are legal in the defense example above.

### 3. Tic-tac-toe

- **Board encodings:** each of 9 squares is empty, X or O: 3^9 = **19,683**. This is an **upper bound**, not the number of reachable positions.
- **Reachable positions:** **5,478**. Two reasons why this is smaller:
  1. X moves first and the players alternate, so the number of X must be equal to the number of O, or one more. A board with five X and no O is possible as an encoding but never happens.
  2. Play stops at the first win. Boards where both players have a line, or where moves continue after a line, never happen.
- **Action labels:** 9 (one for each square).
- **Example board** `X O X / . O . / . . .`: X has 2 and O has 2, so it is **X's turn**. The empty squares are the legal moves: **5** legal placements.
- **Does the board decide the turn?** Yes. With these rules, the counts of X and O tell whose turn it is. So multiplying every encoding by 2 does **not** count legal states: the second copy of each board has the wrong player to move.

### 4. A small Nim game (piles 2, 3, 4)

- **Pile configurations:** each pile can have 0 up to its start size: 3 · 4 · 5 = **60**. With "player to move" (2 players): 60 · 2 = **120** encodings. This is an **encoding count**, not a count of reachable states. For example (1, 3, 4) can only be reached after exactly one move (remove 1 stone from the first pile), so only with player 2 to move. The encoding with player 1 to move is not reachable.
- **Actions** are written as (pile, number removed):
  - at (2, 3, 4): pile 1 has 2 choices, pile 2 has 3, pile 3 has 4, so 2 + 3 + 4 = **9** actions;
  - at (0, 1, 2): pile 2 has 1 choice, pile 3 has 2, so **3** actions;
  - at the terminal position (0, 0, 0): **0** actions.
- **Why "choose a pile" is not enough:** the move also needs the number of stones. The same pile with 1 or 2 stones removed leads to different positions.
- **Unlabeled piles:** if we do not care which pile is which, the configurations (2,3,0) and (3,2,0) are the same. Then there are **28** configurations (sorted triples a ≤ b ≤ c with a ≤ 2, b ≤ 3, c ≤ 4), not 60.

### Comparison table

| Game | Representation | State-count formula | Type of count | Action representation | Legal actions in the example | Hidden information |
| --- | --- | --- | --- | --- | --- | --- |
| Blackjack | Observation: player sum 12–21, dealer showing 1–10, usable ace | 10 · 10 · 2 = 200 (Q-table: 200 · 2 = 400) | Exact for the restricted table; not the whole state space | 2 labels: stick, hit | 2 (both are legal in a non-final decision) | Dealer's hidden card, future cards |
| Durak | Player's hand (one part of the state) and the candidate opponent hands | C(36,6) = 1,947,792 hands; C(29,6) = 475,020 opponent hands | Exact for these parts only; the full state is larger (stock order, table, discard, phase, turn) | 36 "play card" labels + pick up + end attack = 38 | 4 (A♣, 7♥, Q♥, pick up) | Opponent's hand, order of the stock |
| Tic-tac-toe | Board, 9 squares with 3 values each | 3^9 = 19,683 | Upper bound; 5,478 positions are reachable. The board decides the turn. | 9 labels (one per square) | 5 | None |
| Nim (2,3,4) | Three labeled piles + player to move | 3 · 4 · 5 = 60 configurations; 60 · 2 = 120 encodings | Encoding count (upper bound); not all are reachable | 9 labels (pile, number removed) | 9 at (2,3,4); 3 at (0,1,2); 0 at the end | None |

**Which games need a very large Q-table?** Durak is by far the largest. The state has two hands, the table, the stock order and more, and there are 38 action labels. A table with one entry for each (state, action) is not possible. Tic-tac-toe is small (at most 19,683 · 9 = 177,147 entries, and fewer are reachable). Nim has at most 120 · 9 = 1,080 entries. The restricted Blackjack table has 400.

**Why the number of action labels can be different from the number of legal actions:** the labels are all moves that the encoding can express. In one position only some of them are allowed. In tic-tac-toe there are 9 labels, but in the example board only 5 squares are empty. In Durak there are 38 labels, but only 4 are legal in the defense example. The learner needs a fixed set of labels for its table, and it needs the legal actions to choose from.

## Part B — Paper calculations

Original photos, taken before I wrote any learning code:

- [`paper/page1-partA.png`](paper/page1-partA.png) — Part A, games 1–3
- [`paper/page2-partA-partB.png`](paper/page2-partA-partB.png) — Part A, Nim; Part B, model-based and Monte Carlo
- [`paper/page3-partB-td-q.png`](paper/page3-partB-td-q.png) — Part B, TD(0) and Q-learning

The numbers in my paper work agree with the program in Part C, so no result needed a correction.

Policy for methods 1–3: hit at A, C, D; stick at B, E (hit below 20, stick at 20 or 21). Games: game 1 = A→B→END (+1), game 2 = A→END (−1), game 3 = C→D→E→END (0), game 4 = A→B→END (−1). Below is a short copy of the calculations; the full work is in the photos.

### 1. Model-based learning

Counts: N(A,hit) = 3, N(B,stick) = 2, N(C,hit) = N(D,hit) = N(E,stick) = 1.

- T(A,hit,B) = 2/3, reward 0.
- T(A,hit,END) = 1/3, reward −1.
- T(B,stick,END) = 2/2 = 1, mean reward (+1 − 1)/2 = 0.
- T(C,hit,D) = T(D,hit,E) = T(E,stick,END) = 1, reward 0.
- Unobserved pairs: (A,stick), (B,hit), (C,stick), (D,stick), (E,hit).

Policy evaluation, V₀ = 0, V_{k+1}(s) = Σ T · (R + V_k(s')), using only the previous sweep:

| Sweep | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- |
| V₁ | 2/3·(0+0) + 1/3·(−1) = −1/3 | 0 | 0 | 0 | 0 |
| V₂ | 2/3·(0+V₁(B)=0) + 1/3·(−1) = −1/3 | 0 | 0 | 0 | 0 |

**Why sample probabilities need not be the true probabilities.** They come from only 3 games from A. In a fair game with unlimited cards, hitting at 18 busts with any card from 4 to 10, which is 10 of 13 ranks (about 77%). Our sample shows 1 bust in 3 (33%). The four games are constructed examples, not a fair sample of the real game, and a small sample can be far from the truth.

### 2. Direct evaluation (every-visit Monte Carlo, one pass)

| Game | Visit → return |
| --- | --- |
| 1 | A: +1, B: +1 |
| 2 | A: −1 |
| 3 | C: 0, D: 0, E: 0 |
| 4 | A: −1, B: −1 |

| State | Returns | Count | Average |
| --- | --- | --- | --- |
| A | +1, −1, −1 | 3 | −1/3 |
| B | +1, −1 | 2 | 0 |
| C, D, E | 0 | 1 each | 0 |

### 3. TD(0), α = 0.1, γ = 1

Rule: V_new = V + 0.1 · (r + V' − V), with V' = 0 at END. All 16 updates (two passes over games 1–4, values kept between games and passes):

| Pass | Game | Step | State | Old V | Reward | V(next) | Target | New V |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | A | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 2 | B | 0 | 1 | 0 | 1 | 0.1 |
| 1 | 2 | 1 | A | 0 | -1 | 0 | -1 | -0.1 |
| 1 | 3 | 1 | C | 0 | 0 | 0 | 0 | 0 |
| 1 | 3 | 2 | D | 0 | 0 | 0 | 0 | 0 |
| 1 | 3 | 3 | E | 0 | 0 | 0 | 0 | 0 |
| 1 | 4 | 1 | A | -0.1 | 0 | 0.1 | 0.1 | -0.08 |
| 1 | 4 | 2 | B | 0.1 | -1 | 0 | -1 | -0.01 |
| 2 | 1 | 1 | A | -0.08 | 0 | -0.01 | -0.01 | -0.073 |
| 2 | 1 | 2 | B | -0.01 | 1 | 0 | 1 | 0.091 |
| 2 | 2 | 1 | A | -0.073 | -1 | 0 | -1 | -0.1657 |
| 2 | 3 | 1 | C | 0 | 0 | 0 | 0 | 0 |
| 2 | 3 | 2 | D | 0 | 0 | 0 | 0 | 0 |
| 2 | 3 | 3 | E | 0 | 0 | 0 | 0 | 0 |
| 2 | 4 | 1 | A | -0.1657 | 0 | 0.091 | 0.091 | -0.14003 |
| 2 | 4 | 2 | B | 0.091 | -1 | 0 | -1 | -0.0181 |


Final: A = −0.14003, B = −0.0181, C = D = E = 0.

### 4. Q-learning, α = 0.1, γ = 1

Rule: Q_new = Q + 0.1 · (r + max over hit and stick of Q(s',·) − Q). The maximum is taken over **both** actions, also over the untried one (it is 0). Fresh zero table. All 16 updates:

| Pass | Game | Step | State, action | Old Q | Reward | max Q(next) | Target | New Q |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | A, hit | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 2 | B, stick | 0 | 1 | 0 | 1 | 0.1 |
| 1 | 2 | 1 | A, hit | 0 | -1 | 0 | -1 | -0.1 |
| 1 | 3 | 1 | C, hit | 0 | 0 | 0 | 0 | 0 |
| 1 | 3 | 2 | D, hit | 0 | 0 | 0 | 0 | 0 |
| 1 | 3 | 3 | E, stick | 0 | 0 | 0 | 0 | 0 |
| 1 | 4 | 1 | A, hit | -0.1 | 0 | 0.1 | 0.1 | -0.08 |
| 1 | 4 | 2 | B, stick | 0.1 | -1 | 0 | -1 | -0.01 |
| 2 | 1 | 1 | A, hit | -0.08 | 0 | 0 | 0 | -0.072 |
| 2 | 1 | 2 | B, stick | -0.01 | 1 | 0 | 1 | 0.091 |
| 2 | 2 | 1 | A, hit | -0.072 | -1 | 0 | -1 | -0.1648 |
| 2 | 3 | 1 | C, hit | 0 | 0 | 0 | 0 | 0 |
| 2 | 3 | 2 | D, hit | 0 | 0 | 0 | 0 | 0 |
| 2 | 3 | 3 | E, stick | 0 | 0 | 0 | 0 | 0 |
| 2 | 4 | 1 | A, hit | -0.1648 | 0 | 0.091 | 0.091 | -0.13922 |
| 2 | 4 | 2 | B, stick | 0.091 | -1 | 0 | -1 | -0.0181 |


Final Q-values and greedy actions:

| State | Q(stick) | Q(hit) | Greedy action |
| --- | --- | --- | --- |
| A | 0 | −0.13922 | stick |
| B | −0.0181 | 0 | hit |
| C, D, E | 0 | 0 | tie |

**Why an untried zero does not show that the action is good.** The zero is only the start value, not a measurement. At A the greedy action is "stick" only because stick was never tried and still has its start value 0, which is higher than the measured Q(A,hit) = −0.139. At B the greedy action is "hit" because hit was never tried and 0 is higher than −0.0181. In a real game, hitting at 20 (B) busts with any card above an ace, which is 12 of 13 ranks. So hit at B is really a very bad action. We have no evidence about it.

### Why the four final tables are different

- The methods use the data in different ways. Model-based learning builds a model first and then runs 2 sweeps. Monte Carlo only uses full-game results and runs 1 pass. TD(0) and Q-learning move a small step (10%) after each transition and run 2 passes.
- Q-learning estimates Q-values for actions, not values of one fixed policy.
- All tables start from 0, and the sample has only 4 games. For example, V(A) is −1/3 for model-based and Monte Carlo, but −0.14003 for TD(0), and Q(A,hit) is −0.13922. The numbers would come closer with more data and more passes.
- A fixed replay checks the update rules. It does **not** test action selection or exploration, because the actions are given by the histories.

## Part C — Reproduce the calculations

Command: `python -m blackjack_rl.paper_replay`. Outputs are in [`outputs/paper-replay/`](outputs/paper-replay/):

- full printed output: [`replay_output.txt`](outputs/paper-replay/replay_output.txt)
- counts, probabilities, rewards: [`model.csv`](outputs/paper-replay/model.csv); the two sweeps: [`value_sweeps.csv`](outputs/paper-replay/value_sweeps.csv)
- Monte Carlo: [`mc_returns.csv`](outputs/paper-replay/mc_returns.csv), [`mc_summary.csv`](outputs/paper-replay/mc_summary.csv)
- every TD(0) update: [`td_updates.csv`](outputs/paper-replay/td_updates.csv)
- every Q-learning update and final table: [`q_updates.csv`](outputs/paper-replay/q_updates.csv), [`q_final.csv`](outputs/paper-replay/q_final.csv)
- paper versus program: [`paper_vs_program.md`](outputs/paper-replay/paper_vs_program.md)

The program reads exactly `data/paper-games.csv` and `data/states.csv`. It does not deal new cards.

**Result: 49 of 49 values match.** The paper values are typed in `paper/paper_results.json` (copied from my photos). The "difference" is smaller than 10⁻¹² in all rows. The program keeps full precision; on paper I rounded to at most 5 decimals (for example −0.14003). So no discrepancy has to be explained and no paper result needed a correction.

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
|---|---|---|---|---|---|---|
| model-based | N(A,hit) | counts | 3 | 3 | 0 | match |
| model-based | N(B,stick) | counts | 2 | 2 | 0 | match |
| model-based | T(A,hit,B) | model | 2/3 | 0.666667 | 0 | match |
| model-based | R(A,hit,B) | model | 0 | 0 | 0 | match |
| model-based | T(A,hit,END) | model | 1/3 | 0.333333 | 0 | match |
| model-based | R(A,hit,END) | model | -1 | -1 | 0 | match |
| model-based | T(B,stick,END) | model | 1 | 1 | 0 | match |
| model-based | R(B,stick,END) | model | 0 | 0 | 0 | match |
| model-based | V(A) | sweep 1 | -1/3 | -0.333333 | 0 | match |
| model-based | V(B) | sweep 1 | 0 | 0 | 0 | match |
| model-based | V(C) | sweep 1 | 0 | 0 | 0 | match |
| model-based | V(D) | sweep 1 | 0 | 0 | 0 | match |
| model-based | V(E) | sweep 1 | 0 | 0 | 0 | match |
| model-based | V(A) | sweep 2 | -1/3 | -0.333333 | 0 | match |
| model-based | V(B) | sweep 2 | 0 | 0 | 0 | match |
| model-based | V(C) | sweep 2 | 0 | 0 | 0 | match |
| model-based | V(D) | sweep 2 | 0 | 0 | 0 | match |
| model-based | V(E) | sweep 2 | 0 | 0 | 0 | match |
| monte-carlo | average(A) | one pass | -1/3 | -0.333333 | 0 | match |
| monte-carlo | count(A) | one pass | 3 | 3 | 0 | match |
| monte-carlo | average(B) | one pass | 0 | 0 | 0 | match |
| monte-carlo | count(B) | one pass | 2 | 2 | 0 | match |
| monte-carlo | average(C) | one pass | 0 | 0 | 0 | match |
| monte-carlo | count(C) | one pass | 1 | 1 | 0 | match |
| monte-carlo | average(D) | one pass | 0 | 0 | 0 | match |
| monte-carlo | count(D) | one pass | 1 | 1 | 0 | match |
| monte-carlo | average(E) | one pass | 0 | 0 | 0 | match |
| monte-carlo | count(E) | one pass | 1 | 1 | 0 | match |
| td0 | V(B) | pass 1 game 1 step 2 | 0.1 | 0.1 | 0 | match |
| td0 | V(A) | pass 1 game 2 step 1 | -0.1 | -0.1 | 0 | match |
| td0 | V(A) | pass 1 game 4 step 1 | -0.08 | -0.08 | 0 | match |
| td0 | V(B) | pass 1 game 4 step 2 | -0.01 | -0.01 | 0 | match |
| td0 | V(A) | pass 2 game 1 step 1 | -0.073 | -0.073 | 0 | match |
| td0 | V(B) | pass 2 game 1 step 2 | 0.091 | 0.091 | 0 | match |
| td0 | V(A) | pass 2 game 2 step 1 | -0.1657 | -0.1657 | 0 | match |
| td0 | V(A) | pass 2 game 4 step 1 | -0.14003 | -0.14003 | 0 | match |
| td0 | V(B) | pass 2 game 4 step 2 | -0.0181 | -0.0181 | 0 | match |
| td0 | V(C) | final | 0 | 0 | 0 | match |
| td0 | V(D) | final | 0 | 0 | 0 | match |
| td0 | V(E) | final | 0 | 0 | 0 | match |
| q-learning | Q(B,stick) | pass 1 game 1 step 2 | 0.1 | 0.1 | 0 | match |
| q-learning | Q(A,hit) | pass 1 game 2 step 1 | -0.1 | -0.1 | 0 | match |
| q-learning | Q(A,hit) | pass 1 game 4 step 1 | -0.08 | -0.08 | 0 | match |
| q-learning | Q(B,stick) | pass 1 game 4 step 2 | -0.01 | -0.01 | 0 | match |
| q-learning | Q(A,hit) | pass 2 game 1 step 1 | -0.072 | -0.072 | 0 | match |
| q-learning | Q(B,stick) | pass 2 game 1 step 2 | 0.091 | 0.091 | 0 | match |
| q-learning | Q(A,hit) | pass 2 game 2 step 1 | -0.1648 | -0.1648 | 0 | match |
| q-learning | Q(A,hit) | pass 2 game 4 step 1 | -0.13922 | -0.13922 | 0 | match |
| q-learning | Q(B,stick) | pass 2 game 4 step 2 | -0.0181 | -0.0181 | 0 | match |


## Part D — Simulated games

### Fixed-policy value estimates

For seeds 11, 22 and 33 I played 10,000 games each with the policy "hit below 20, stick at 20 or 21". Every step is saved in the game logs. Then the **same saved games in the same order** were given to the three methods. Terminal transitions are converted to `END` for the learners.

- Game logs: [`seed_11/games.csv`](outputs/fixed-policy/seed_11/games.csv), [`seed_22/games.csv`](outputs/fixed-policy/seed_22/games.csv), [`seed_33/games.csv`](outputs/fixed-policy/seed_33/games.csv) (columns: seed, game, step, observation, action, reward, next observation, terminal flag, cut-off flag).
- Estimates after 100, 1,000 and 10,000 games, with visit counts: `seed_*/estimates.csv`.
- Value tables by player sum and dealer card, separate for usable ace yes/no, with visit counts and `-` for missing estimates: `seed_*/tables.md`.
- Model-based convergence: [`model_based_convergence.csv`](outputs/fixed-policy/model_based_convergence.csv). Differences between methods: [`method_differences.md`](outputs/fixed-policy/method_differences.md). Frequent and rare states: [`state_comparison.csv`](outputs/fixed-policy/state_comparison.csv).

**Model-based settings:** tolerance 0.000001, maximum 10,000 sweeps. All 9 runs converged:

| Seed | Games | Sweeps | Converged | Last max change | States with a value |
| --- | --- | --- | --- | --- | --- |
| 11 | 100 | 6 | yes | 0 | 96 |
| 11 | 1,000 | 12 | yes | 2.2e-10 | 243 |
| 11 | 10,000 | 11 | yes | 3.4e-07 | 280 |
| 22 | 100 | 6 | yes | 0 | 116 |
| 22 | 1,000 | 10 | yes | 5.0e-07 | 245 |
| 22 | 10,000 | 11 | yes | 9.1e-08 | 280 |
| 33 | 100 | 5 | yes | 0 | 118 |
| 33 | 1,000 | 11 | yes | 1.3e-07 | 247 |
| 33 | 10,000 | 11 | yes | 1.8e-07 | 280 |

**Checks.** No game was cut off at the 100-action limit (`cutoffs.json` has empty lists). In every log the policy was followed, rewards of non-final steps are 0, and the states in the chain match. I also ran the learners again on the saved game logs, and the estimates are identical to `estimates.csv`.

**Missing estimates and how I handled them.**

- A state that was never visited has **no estimate**. It is shown as `-` in the tables. I do not write a zero there, and I do not use the game rules to fill it.
- Model-based: a value is computed only when the pair (state, policy action) was observed. If a next state had no observed pair, its value would be taken as 0 and the states that depend on it would be marked "affected". This **never happened** in my runs (0 affected states in all 9 runs), because the policy was followed in every game and no game was cut off.
- TD(0): a value is shown only after at least one update. The start value 0 is not shown as an estimate.
- Coverage grows with more games: after 100 games only 96–118 of the 280 possible states were visited, after 1,000 games 243–247, and after 10,000 games all 280 (180 without usable ace, 100 with usable ace) in every seed.

**Examples of frequent and rare states (seed 11).** Values: model-based / Monte Carlo / TD(0). Format: estimate (visits).

| State (sum, dealer, usable ace) | 100 games | 1,000 games | 10,000 games |
| --- | --- | --- | --- |
| (20, 10, no) frequent | 0.333 / 0.333 / 0.173 (6) | 0.467 / 0.467 / 0.511 (60) | 0.416 / 0.416 / 0.439 (606) |
| (19, 10, no) frequent | −0.295 / −0.286 / −0.279 (7) | −0.665 / −0.689 / −0.775 (61) | −0.729 / −0.736 / −0.723 (447) |
| (18, 10, no) frequent | −0.178 / 0.000 / −0.161 (6) | −0.647 / −0.581 / −0.819 (43) | −0.759 / −0.737 / −0.943 (415) |
| (4, 6, no) rare | no estimate (0) | −0.620 / −1.000 / 0.000 (1) | −0.693 / −1.000 / 0.000 (1) |
| (4, 1, no) rare | no estimate (0) | no estimate (0) | −0.698 / −1.000 / −0.088 (2) |
| (12, 4, yes) rare | no estimate (0) | no estimate (0) | −0.666 / −1.000 / −0.074 (1) |

Average size of the difference between two methods (root mean square, 10,000 games, from `method_differences.md`):

| Seed | States | Model-based vs Monte Carlo | Model-based vs TD(0) | Monte Carlo vs TD(0) |
| --- | --- | --- | --- | --- |
| 11 | 147 frequent (30+ visits) | 0.051 | 0.111 | 0.125 |
| 22 | 148 frequent | 0.061 | 0.110 | 0.135 |
| 33 | 151 frequent | 0.061 | 0.103 | 0.120 |
| 11 / 22 / 33 | 20 / 11 / 14 rare (1–5 visits) | 0.514 / 0.499 / 0.403 | 0.420 / 0.389 / 0.324 | 0.676 / 0.584 / 0.492 |

**Explanation.** There is no exact reference value, so I can only compare the methods with each other and with more data.

- **More games help.** For frequent states the three estimates come close to each other when the number of visits grows, for example (19, 10, no): −0.729 / −0.736 / −0.723 after 10,000 games. After 100 games the same state has 7 visits, and the estimates are much less reliable. In states 20 and 21 (stick) model-based and Monte Carlo are exactly the same, because both are just the mean reward.
- **Model-based and Monte Carlo** are close for frequent states (RMS 0.05–0.06). They differ because the model-based method also uses what it learned about the next states, while Monte Carlo only averages the returns that really followed this state.
- **TD(0)** is further from both (RMS about 0.11). It uses a constant learning rate of 0.1, so each value mainly remembers the last 10 or so visits. It is noisy. For example (18, 10, no) is −0.943 for TD(0) but −0.737 for Monte Carlo.
- **Rare states are different for each method.** With 1 visit Monte Carlo is just one result (+1 or −1). TD(0) moves only a small step from its start value, so it stays near 0. Model-based uses the values of the next states, which have many visits, so it gives a more careful number (about −0.69 for (4, 6, no) at 10,000 games). I cannot say which of the numbers is the true one, only that all three have very little data there.

### Q-learning and baseline evaluation

Setup: 50,000 training games for each seed (11, 22, 33), learning rate 0.1, discount 1.0, epsilon 0.2 or 0. Exploration chooses one of the two actions with equal probability. Ties are broken at random with a seeded generator. The trained table is frozen and evaluated with greedy actions on 10,000 new games, with evaluation seeds 1011, 1022, 1033. The baseline uses the same evaluation seed. No game was cut off, in training or in evaluation (`training_cutoffs.json` is empty).

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | Baseline (hit below 20) | 1011 | 2962 | 6462 | 576 | 0 | −0.3500 |
| 11 | Q-learning, ε = 0 | 1011 | 4102 | 4900 | 998 | 0 | −0.0798 |
| 11 | Q-learning, ε = 0.2 | 1011 | 4204 | 4894 | 902 | 0 | −0.0690 |
| 22 | Baseline (hit below 20) | 1022 | 2935 | 6481 | 584 | 0 | −0.3546 |
| 22 | Q-learning, ε = 0 | 1022 | 4161 | 4909 | 930 | 0 | −0.0748 |
| 22 | Q-learning, ε = 0.2 | 1022 | 4209 | 4847 | 944 | 0 | −0.0638 |
| 33 | Baseline (hit below 20) | 1033 | 2950 | 6496 | 554 | 0 | −0.3546 |
| 33 | Q-learning, ε = 0 | 1033 | 4227 | 4877 | 896 | 0 | −0.0650 |
| 33 | Q-learning, ε = 0.2 | 1033 | 4178 | 4941 | 881 | 0 | −0.0763 |

Mean over the three seeds: baseline −0.353, Q-learning ε = 0.2 −0.070, Q-learning ε = 0 −0.073. Full table: [`outputs/q-learning/evaluation_summary.md`](outputs/q-learning/evaluation_summary.md).

### Plots and learned policies

- Training curves (1,000-game blocks) and evaluation points, both epsilons, three seeds: [`outputs/q-learning/training_curves.png`](outputs/q-learning/training_curves.png). Lines are training; the circles and squares in the grey area are the evaluation.
- For each seed and epsilon, in `outputs/q-learning/seed_<seed>/eps_<0.2 or 0>/`:
  - `q_table.csv`: Q-values, visit counts of both actions and the greedy action;
  - `greedy_policy_ace0.csv`, `greedy_policy_ace1.csv`: hit/stick tables without and with usable ace;
  - `visit_counts_ace0.csv`, `visit_counts_ace1.csv`: visits as stick/hit;
  - `tables.md`: the same in readable form; `greedy_policy.png`: picture of the policy;
  - `training_blocks.csv`, `training_games.csv`, `evaluation.json`.

### Explain your results

**Prediction versus control.** The model-based method (with policy evaluation), Monte Carlo and TD(0) are *prediction* methods here: they evaluate one given policy (hit below 20) and learn state values V. Q-learning is a *control* method: it learns action values Q and chooses actions itself. Model-based learning learns a model (T and R), Monte Carlo learns state values from full returns, TD(0) learns state values from single steps, and Q-learning learns action values from single steps.

**Terminal updates.** At the end of a game the next state is `END` and its value is 0, so the target is only the reward (+1, 0 or −1). A bust is a terminal hit with reward −1. A hit that does not bust has reward 0, so its target is the value of the next state. This is how the final result moves back to earlier decisions. For Monte Carlo the whole game must finish first, and the return is the final reward.

**Usable aces.** The flag changes the decisions. In the policy of seed 11 with ε = 0.2, a hard 15 sticks against dealer cards 2–6, but a soft 15 (with a usable ace) hits against every dealer card. A soft hand cannot bust with one more card, so hitting is cheaper. Hard totals of 18 and more always stick, but a soft 18 hits against most dealer cards. Usable-ace states are also visited less often (100 states instead of 180, and lower counts), so their numbers are noisier.

**Exploration.**
- With ε = 0 the learner only repeats what looks best. In seed 11, 16 states have one action that was **never tried**. In one of them the result is clearly wrong: soft 21 against dealer 4 is "hit", because stick was never tried and kept its start value 0, while hit got a positive Q-value (0.278) from early results. In the ε = 0.2 run of the same seed, the Q-value of stick in this state is 0.981, so stick is much better. In the same seed with ε = 0.2, only 1 state has an untried action (soft 12 against dealer 6, stick), and soft 21 against dealer 4 is "stick". So exploration fixed this problem.
- But exploration did not clearly give a better policy: ε = 0.2 was better than ε = 0 on seeds 11 and 22 and worse on seed 33, each time by about 0.01. One evaluation of 10,000 games has a standard error of about 0.01, so this difference is not clear. The mean over three seeds is −0.070 for ε = 0.2 and −0.073 for ε = 0.
- The training curve of ε = 0.2 stays near −0.14 (seed 11, last 10 blocks) while its evaluation is −0.069. The reason is that 20% of the training moves are random, so training and evaluation measure different things. For ε = 0 the training curve reaches about −0.06 (seed 11) and is close to its evaluation (−0.080).

**Limited visits and coverage.** The learning rate is constant (0.1), so each Q-value mostly remembers its last 10 or so updates and keeps moving. Some cells are probably noise: for example, with ε = 0.2 in seed 11, hard 17 against dealer 8 is "hit" (Q-values −0.664 for stick and −0.560 for hit, 245 and 200 visits), and the two numbers are close. Rare states, such as low sums, have few visits. The numbers in the tables are estimates, not exact values.

**Variation across seeds.** All three seeds give the same general result (baseline about −0.35, Q-learning about −0.07 to −0.08), but the single numbers differ by up to 0.015 and the training curves are different. The single cells of the learned policies also change from seed to seed and from epsilon to epsilon, so I would not trust one cell. I only inspected the tables of seed 11 in detail.

**Did Q-learning improve on the baseline?** Yes, in all six trained tables (3 seeds × 2 epsilons). The mean reward went from about −0.35 to between −0.064 and −0.080 on new games with the same evaluation seed, which is about 0.28 better per game and much more than the noise of about 0.01. This is not a guarantee: the training was finite (50,000 games), the policy is not perfect (we saw wrong cells), and the result depends on the seed. The game is still a losing game for the player (negative mean reward), as expected.
