// setting max possible goals predicted
// 12 is reasonable
const MAX_GOALS = 12;

// Factorial lookup table, built once
const factorials = [1];
for (let i = 1; i <= MAX_GOALS; i++)
{
    factorials[i] = factorials[i-1] * i;
}

// Poisson probability mass function
// Poisson regresion formula: P(x=k) = e^-lambda*lambda^k / k!
// lambda is the mean
// k is the event you are looking for x to reach (in this case the num of goals)
function poissonPMF(lambda, k)
{
    let xG = Math.exp(-lambda) * Math.pow(lambda, k) / factorials(k);
    return xG;
}

function predictMatch(model, homeTeam, awayTeam, neutral)
{
    const eloH = model.elos[homeTeam];
    const eloA = model.elos[awayTeam];
    if (eloH === undefined || eloA === undefined) return null;

    // Positive when the home team is stronger
    const eloDiff = (eloH-eloA) / 100;

    // World Cup matches are almost always neutral, so this is usually 0
    const homeAdv = neutral ? 0:1;

    // Log-linear formula from the trained GLM (matches numpy.exp in train_v1.py)
    const lambdaH = Math.exp(
        model.home_model.intercept +
        model.home_model.elo_diff * eloDiff +
        model.home_model.home_adv * homeAdv
    );

    const lambdaA = Math.exp(
        model.away_model.intercept +
        model.away_model.elo_diff_away * (-eloDiff)
    );

    // Per team Poisson probability vectors (for each goal): pH[k] = P(home scores k)
    const pH = [];
    const pA = [];
    for (let k = 0; k <= MAX_GOALS; k++)
    {
        pH.push(poissonPMF(lambdaH, k));
        pA.push(poissonPMF(lambdaA, k))
    }

    // Full joint matrix
    // Rows home goal probability, columns away goal probaility
    const matrix = [];
    let homeWin = 0;
    let draw = 0;
    let awayWin = 0;

    for (let i = 0; i <= MAX_GOALS; i++)
    {
        matrix[i] = [];
        for (let j = 0; j <= MAX_GOALS; j++)
        {
            const p = pH[i]*pH[j];
            matrix[i][j] = p;

            if (i> j)
            {
                homeWin+=p;
            }
            else if (i == j)
            {
                draw+=p;
            }
            else{
                awayWin+=p;
            }
        }
    }

    // Rank every scoreline worth showing (>0.1% probability), keep top 10
    const scores = [];
    for (let i = 0; i <= MAX_GOALS; i++)
    {
        for (let j = 0; j <= MAX_GOALS; j++)
        {
            if (matrix[i][j]> 0.001)
            {
                scores.push({ h: i, a: j, p: matrix[i][j] });
            }
        }
    }
    scores.sort((a, b)=> b.p-a.p);

    return {
        lambdaH,
        lambdaA,
        homeWin,
        draw,
        awayWin,
        matrix,
        topScores: scores.slice(0, 10),
  };

}

// DOM helpers

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

// Populate team selector dropdowns

function populateDropdowns(teams)
{
    const homeSel = $('#home-team');
    const awaySel = $('#away-team');
    homeSel.innerHTML = '';
    awaySel.innerHTML = '';
    teams.forEach(t=> {
        homeSel.appendChild(new Option(t, t));
        awaySel.appendChild(new Option(t, t));
    });

    //  Defaults so the page isn't blank on load
    homeSel.value = 'Argentina';
    awaySel.value = 'Portugal';
}

// This function will render prediction results into the DOM
function renderResults(model, result)
{
    const panel = $('.results-panel');

    if (!result)
    {
        panel.classList.remove('visible');
        return;
    }
    panel.classList.add('visible');

    // Expected goals
    $('#home-team-name').textContent = $('#home-team').value;
    $('#away-team-name').textContent = $('#away-team').value;
    $('#home-lambda').textContent = result.lambdaH.toFixed(3);
    $('#away-lambda').textContent = result.lambdaA.toFixed(3);

    // W/D/L bars, width proportinal to probability
    const total = result.homeWin + result.draw + result.awayWin;
    const hPct = (result.homeWin / total * 100).toFixed(1);
    const dPct = (result.draw / total * 100).toFixed(1);
    const aPct = (result.awayWin / total * 100).toFixed(1);

    const winBar = $('#win-bar');
    const drawBar = $('#draw-bar');
    const lossBar = $('#loss-bar');
    
    winBar.style.width = hPct + '%';
    winBar.textContent = hPct + '%';
    drawBar.style.width = dPct + '%';
    drawBar.textContent = dPct + '%';
    lossBar.style.width = aPct + '%';
    lossBar.textContent = aPct + '%';
    
    $('#win-pct').textContent = hPct + '%';
    $('#draw-pct').textContent = dPct + '%';
    $('#loss-pct').textContent = aPct + '%';
    
    // Score matrix (home x away, 0..6), green/amber/red by outcome
    const tbody = $('#matrix-body');
    tbody.innerHTML = '';

    // Header row
    let headerRow = '<tr><th></th>';
    for (let j = 0; j<=6; j++)
    {
        headerRow += '<th>${j}</th>'
    }
    headerRow += '</tr>';
    tbody.innerHTML += headerRow;

    // One row per home goal count
    for (let i = 0; i<=6; i++)
    {
        let row = '<tr><th>${i}</th>';
        for (let j = 0; j <=6; j++)
        {
            const p = result.matrix[i][j];
            const cls = i > j ? 'win' : i==j ? 'draw' : 'loss';
            const pct = (p*100).toFixed(1);
            if (p > 0.001) {
                row += '<td class="cell ${cls}">${pct}%</td>';
            }
            else
            {
                row += `<td class="cell">—</td>`; // if less than 0.01, it is negligigble
            }
        }
        row += '</tr>';
        tbody.innerHTML += row;
    }

    // Top 10 scorelines
    const scoreList = $('score-list');
    scoreList.innerHTML = '';
    result.topScores.forEach(s => {
        const pct = (s.p * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'score-item';
        div.innerHTML =
            '<span class="sc">${s.h} - ${s.a} </span>' +
            '<span class="prob">${pct}%</span>';
        scoreList.appendChild(div);
    });
}

// App entry point
(function init() {
  const loading = $('.loading');
  const errorDiv = $('.error');
  const predictBtn = $('#predict-btn');
 
  // model.js failed to load or set MODEL_DATA
  if (typeof MODEL_DATA === 'undefined' || !MODEL_DATA) {
    loading.style.display = 'none';
    errorDiv.style.display = 'block';
    errorDiv.textContent =
      'Failed to load the prediction model. ' +
      'Make sure model.js is deployed alongside this page.';
    console.error('init error: MODEL_DATA is undefined — missing model.js');
    return;
  }
 
  const model = MODEL_DATA;
 
  // Loading → ready
  loading.style.display = 'none';
  errorDiv.style.display = 'none';
  predictBtn.disabled = false;
 
  populateDropdowns(model.teams);
 
  predictBtn.addEventListener('click', () => {
    const home = $('#home-team').value;
    const away = $('#away-team').value;
    const neutral = $('#neutral-toggle').checked;
 
    if (home === away) {
      renderResults(model, null);
      alert('Please select two different teams.');
      return;
    }
 
    const result = predictMatch(model, home, away, neutral);
    renderResults(model, result);
  });
 
  // Run once on load so the page isn't empty — Argentina vs Brazil,
  // neutral venue, is a reasonable default group-stage matchup.
  predictBtn.click();
})();
