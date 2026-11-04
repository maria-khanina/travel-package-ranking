# Travel Package Ranking

A travel site shows each search a short list of packages (flight, hotel, room and board). This project ranks the candidate packages of a search by the chance that the traveler books them, and then chooses the margin of each shown package so that the expected profit is highest. Because the true booking behaviour is known only in a simulation, the market is simulated: the true choice model is hidden from the ranker and used only to grade it.

## The simulated market

- **World.** A catalogue of destinations, hotels of ten types (beach resort, family club, spa retreat, city hotel, ski lodge and others), room types, boards and flights from US hubs, and 100,000 search sessions, each with 30 candidate packages (3 million rows). Every traveler has visible attributes (origin airport, budget, trip details) and 14 hidden tastes that the ranker never sees.
- **Choice.** The traveler books one package or none by a discrete choice (random utility) model: the utility of a package combines its fit to the hidden tastes, the hotel quality, the amenities (distance to the beach or the centre, kids club, pool, spa, transfer, balcony, room size), the price against the budget and random noise; four amenities have no effect at all, as distractors. About 13.5% of the sessions end with a booking.
- **Texts.** Hotel and room descriptions are written by Llama-3.1-8B-AWQ and the free-text search queries by Qwen3-4B-AWQ, both served locally with vLLM, so that the ranker has to read the tastes out of text, as on a real site.

## Pipeline

1. `world`, `texts`, `simulate`: build the catalogue and the sessions, generate the texts, simulate the bookings.
2. `features`: tabular features (price, quality, amenities, trip), embeddings of the query and the hotel text (Qwen3-Embedding-0.6B and BGE are compared), their cosine similarities, and taste labels: the similarity of the query to characteristic labels of hotels and rooms, combined with the labels each package actually has.
3. `train`: CatBoost on GPU, pointwise classification against two ranking losses (YetiRank and softmax), with early stopping on a separate stopping set; 14 feature and loss setups are compared.
4. `calibrate`: Platt and isotonic calibration of the booking probabilities.
5. `explain`: SHAP values by feature group.
6. `margin`: for each session, the slate of 3 packages and the margin of each one (5% to 25%) that maximise the expected profit under the predicted choice probabilities.
7. `bootstrap`, `truth-metrics`, `cold-report`: confidence intervals, metrics against the true probabilities, results for hotels unseen in training.

## Evaluation protocol

- 20% of the sessions are a closed test set that no choice was made on; all numbers below are from the other 80% (79,987 sessions, 10,798 bookings).
- 5-fold cross-validation grouped by session, so the packages of one search never fall into training and validation at once.
- 15% of the hotels are cold: excluded from all training and scored separately, as new hotels on a real site.
- 95% intervals come from 2,000 bootstrap resamples of the sessions; differences between models are tested with paired resamples.

## Results

**Ranking** (cross-validation; hit rate at 3 is the share of booked sessions whose booked package is among the top 3):

| Model | Hit rate at 3 (95% interval) | NDCG at 10 | Hit rate at 3, cold hotels |
|---|---|---|---|
| random order | 0.100 | 0.151 | |
| tabular features without amenities | 0.307 | 0.361 | 0.245 |
| tabular features | 0.333 | 0.382 | 0.289 |
| + text similarities and taste labels | 0.342 (0.334 to 0.351) | 0.389 | 0.276 to 0.311 |

- The amenities are the largest single gain (+0.026 in hit rate at 3); text similarities and taste labels add about one more point, mostly for cold hotels, where the hotel identity cannot help.
- The two ranking losses and the pointwise loss are equal within the intervals (hit rate at 3 between 0.338 and 0.342).
- Cold hotels are ranked worse than warm ones (hit rate at 3 about 0.29 against 0.35), the expected cost of having no booking history.

**Calibration.** The raw CatBoost probabilities are already close to the observed booking rate (expected calibration error 0.00021); isotonic calibration lowers it to 0.00012 and the calibration error per session from 0.0067 to 0.0053, so the probabilities can be used as prices of risk in the margin step.

**Explanation.** By SHAP, the amenities (20% of the total attribution) and the price (18%) weigh most, then the traveler and trip (17%), the taste-label similarities (13%) and the hotel quality (11%); the four distractor amenities get almost none.

**Margin.** Expected profit per search session, graded with the true choice model (95% intervals):

| Policy | Expected profit per session |
|---|---|
| random packages, logged margins | 19.8 (19.1 to 20.6) |
| ranked by booking probability, logged margins | 61.4 (59.7 to 63.2) |
| ranked by expected profit, logged margins | 75.7 (73.7 to 77.8) |
| best single margin for all packages (20%) | 82.8 (80.5 to 85.0) |
| **optimised margin and slate per session** | **85.2 (82.9 to 87.4)** |
| oracle with the true probabilities | 94.2 (91.7 to 96.7) |

Optimising the margin per package earns 12% more than ranking by expected profit at the logged margins and 3% more than the best single margin, and closes about half of the gap between the ranker at logged margins and the oracle that knows the true probabilities. For the optimised policy the profit predicted by the model (84.4) is close to the true one (85.2), so the estimate can be used before the policy is tried.

## Running

```
pip install -r requirements.txt
python -m src.cli world
bash scripts/vllm_serve.sh <model> <port>   # in WSL; one vLLM server fits an 8 GB GPU
python -m src.cli texts --job <job> --model <model>
python -m src.cli simulate
python -m src.cli features
python -m src.cli train
python -m src.cli calibrate
python -m src.cli explain
python -m src.cli margin
```

All settings (catalogue, choice model, splits, CatBoost parameters, margin grid) are in `configs/`. CatBoost trains on the GPU, so the vLLM server has to be stopped first. The generated data (`data/`) is not in the repository. Tests run with `python -m pytest`; the code is checked with `python -m pycodestyle --max-line-length=88 src configs tests` and `python -m pyright` (strict).
