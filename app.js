
const API = "";

let currentUser = null;
let currentRun = null;

const $ = (id) => document.getElementById(id);

function showMessage(id, text, success = false) {
    const el = $(id);
    if (!el) return;
    el.textContent = text;
    el.className = success ? "message success" : "message";
}

function togglePassword(inputId, button) {
    const input = $(inputId);
    if (!input) return;

    if (input.type === "password") {
        input.type = "text";
        button.textContent = "🙈";
        button.title = "Hide password";
        button.setAttribute("aria-label", "Hide password");
    } else {
        input.type = "password";
        button.textContent = "👁";
        button.title = "Show password";
        button.setAttribute("aria-label", "Show password");
    }
}

function showRegister() {
    $("loginPage").classList.add("hidden");
    $("registerPage").classList.remove("hidden");
}

function showLogin() {
    $("registerPage").classList.add("hidden");
    $("loginPage").classList.remove("hidden");
}

async function registerUser(event) {
    event.preventDefault();

    const name = $("registerName").value.trim();
    const email = $("registerEmail").value.trim();
    const password = $("registerPassword").value;
    const confirm = $("confirmPassword").value;

    if (password !== confirm) {
        showMessage("registerMessage", "Passwords do not match.");
        return;
    }

    if (password.length < 8) {
        showMessage(
            "registerMessage",
            "Password must contain at least 8 characters."
        );
        return;
    }

    try {
        const response = await fetch(`${API}/api/register`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({name, email, password})
        });

        const data = await response.json();

        if (!response.ok) {
            showMessage("registerMessage", data.error || "Registration failed.");
            return;
        }

        showMessage(
            "registerMessage",
            "Account created successfully. Please login.",
            true
        );

        setTimeout(() => {
            $("registerForm").reset();
            showLogin();
        }, 700);

    } catch (error) {
        showMessage(
            "registerMessage",
            "Backend is not running. Start app.py first."
        );
    }
}

async function loginUser(event) {
    event.preventDefault();

    const email = $("loginEmail").value.trim();
    const password = $("loginPassword").value;

    try {
        const response = await fetch(`${API}/api/login`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({email, password})
        });

        const data = await response.json();

        if (!response.ok) {
            showMessage("loginMessage", data.error || "Invalid login.");
            return;
        }

        currentUser = data.user;
        localStorage.setItem("soundAILoggedInUser", JSON.stringify(currentUser));

        showDashboard();
        loadDatasets();
        loadHistory();

    } catch (error) {
        showMessage(
            "loginMessage",
            "Backend is not running. Start app.py first."
        );
    }
}

function showDashboard() {
    $("loginPage").classList.add("hidden");
    $("registerPage").classList.add("hidden");
    $("dashboard").classList.remove("hidden");

    const initial = (currentUser.name || "U").charAt(0).toUpperCase();

    $("topUserName").textContent = currentUser.name;
    $("sidebarName").textContent = currentUser.name;
    $("sidebarEmail").textContent = currentUser.email;
    $("topAvatar").textContent = initial;
    $("sidebarAvatar").textContent = initial;
}

function logout() {
    currentUser = null;
    currentRun = null;
    localStorage.removeItem("soundAILoggedInUser");

    $("dashboard").classList.add("hidden");
    $("loginPage").classList.remove("hidden");
    $("loginForm").reset();
}

function openPage(pageId, button) {
    document.querySelectorAll(".page").forEach(
        page => page.classList.remove("active-page")
    );

    const page = $(pageId);
    if (page) page.classList.add("active-page");

    document.querySelectorAll(".nav").forEach(
        nav => nav.classList.remove("active")
    );

    if (button) button.classList.add("active");

    const titles = {
        home: "Home",
        datasets: "Datasets",
        addDataset: "Add Dataset",
        testing: "Testing",
        performance: "Performance",
        history: "History",
        comparison: "Comparison",
        confusion: "Confusion Matrix"
    };

    $("pageTitle").textContent = titles[pageId] || "Home";

    if (pageId === "history") loadHistory();
}

function openPageByName(pageId) {
    const button = [...document.querySelectorAll(".nav")]
        .find(btn => (btn.dataset.page || "") === pageId);

    openPage(pageId, button);
}

async function loadDatasets() {
    try {
        const response = await fetch(`${API}/api/datasets`);
        const data = await response.json();

        const select = $("testDataset");
        if (!select) return;

        select.innerHTML = "";

        data.forEach(item => {
            const option = document.createElement("option");
            option.value = `${item.equipment}|${item.snr}`;
            option.textContent =
                `${item.equipment} — ${item.snr} ` +
                (item.available ? `(${item.file_count} WAV)` : "(not found)");
            option.disabled = !item.available;
            select.appendChild(option);
        });

        renderDatasetCards(data);

    } catch (error) {
        showMessage(
            "datasetMessage",
            "Could not load datasets. Start the backend."
        );
    }
}

function renderDatasetCards(data) {
    const container = $("datasetCards");
    if (!container) return;

    container.innerHTML = "";

    ["Fan", "Pump"].forEach(machine => {
        const card = document.createElement("div");
        card.className = "dataset-card";

        const items = data.filter(x => x.equipment === machine);

        card.innerHTML = `<h2>${machine}</h2>`;

        items.forEach(item => {
            const status = item.available
                ? `<span class="available">${item.file_count} WAV files</span>`
                : `<span class="not-available">Folder not found</span>`;

            card.innerHTML += `
                <div>
                    <span>${item.snr}</span>
                    ${status}
                </div>
            `;
        });

        container.appendChild(card);
    });
}

async function runAllModels() {
    if (!currentUser) return;

    const value = $("testDataset").value;

    if (!value) {
        showMessage("testMessage", "Please select a dataset.");
        return;
    }

    const [equipment, snr] = value.split("|");

    const button = $("runModelsButton");
    const status = $("testStatus");

    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span> PROCESSING DATASET...`;

    status.className = "run-status visible";
    status.innerHTML = `
        <div class="progress-track">
            <div class="progress-bar"></div>
        </div>
        <div class="progress-text">
            Loading all WAV files → extracting 24 features →
            training/evaluating all five models...
        </div>
    `;

    showMessage("testMessage", "");

    try {
        const response = await fetch(`${API}/api/run-all`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                user_id: currentUser.id,
                equipment,
                snr
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Model execution failed.");
        }

        currentRun = data;

        renderResults(data);
        await loadHistory();

        status.className = "run-status visible success-status";
        status.innerHTML = `
            <strong>✓ Completed</strong>
            <span>${data.file_count} WAV files processed</span>
            <span>${data.failed_files} failed</span>
        `;

        showMessage(
            "testMessage",
            "All five models completed. Actual results are now available.",
            true
        );

        button.innerHTML = `RUN ALL 5 MODELS →`;

    } catch (error) {
        status.className = "run-status visible";
        status.innerHTML = `<strong>Execution stopped:</strong> ${error.message}`;

        showMessage("testMessage", error.message);
        button.innerHTML = `RUN ALL 5 MODELS →`;

    } finally {
        button.disabled = false;
    }
}

function renderResults(data) {
    renderComparison(data.results);
    renderPerformance(data.results);
    renderConfusion(data.results);

    openPageByName("comparison");
}

function renderComparison(results) {
    const body = $("comparisonBody");
    if (!body) return;

    body.innerHTML = "";

    Object.entries(results).forEach(([model, r]) => {
        body.innerHTML += `
            <div class="comparison-row">
                <span class="model-name">${model}</span>
                <span>${r.accuracy.toFixed(2)}%</span>
                <span>${r.precision.toFixed(2)}%</span>
                <span>${r.recall.toFixed(2)}%</span>
                <span>${r.f1.toFixed(2)}%</span>
            </div>
        `;
    });

    const bars = $("comparisonBars");
    if (!bars) return;

    bars.innerHTML = "";

    Object.entries(results).forEach(([model, r]) => {
        bars.innerHTML += `
            <div class="metric-bar-row">
                <div class="metric-bar-label">${model}</div>
                <div class="metric-bar-track">
                    <div class="metric-bar-fill"
                         style="width:${Math.max(0, Math.min(100, r.accuracy))}%">
                    </div>
                </div>
                <div class="metric-bar-value">${r.accuracy.toFixed(2)}%</div>
            </div>
        `;
    });
}

function renderPerformance(results) {
    const container = $("performanceCards");
    if (!container) return;

    container.innerHTML = "";

    Object.entries(results).forEach(([model, r]) => {
        container.innerHTML += `
            <div class="performance-card">
                <div class="performance-card-head">
                    <h3>${model}</h3>
                    <span>Actual run</span>
                </div>
                <div class="metric-grid">
                    <div><small>Accuracy</small><b>${r.accuracy.toFixed(2)}%</b></div>
                    <div><small>Precision</small><b>${r.precision.toFixed(2)}%</b></div>
                    <div><small>Recall</small><b>${r.recall.toFixed(2)}%</b></div>
                    <div><small>F1-Score</small><b>${r.f1.toFixed(2)}%</b></div>
                    <div><small>ROC-AUC</small><b>${r.roc_auc == null ? "—" : r.roc_auc.toFixed(2) + "%"}</b></div>
                </div>
            </div>
        `;
    });
}

function renderConfusion(results) {
    const container = $("confusionContainer");
    if (!container) return;

    container.innerHTML = "";

    Object.entries(results).forEach(([model, r]) => {
        const labels = r.labels || [];
        const cm = r.confusion_matrix || [];

        let html = `
            <div class="confusion-card">
                <div class="performance-card-head">
                    <h3>${model}</h3>
                    <span>Confusion Matrix</span>
                </div>
                <div class="matrix-scroll">
                    <table class="matrix-table">
                        <thead>
                            <tr>
                                <th>Actual \\ Predicted</th>
                                ${labels.map(x => `<th>${x}</th>`).join("")}
                            </tr>
                        </thead>
                        <tbody>
        `;

        labels.forEach((label, i) => {
            html += `<tr><th>${label}</th>`;
            labels.forEach((_, j) => {
                html += `<td>${cm[i]?.[j] ?? 0}</td>`;
            });
            html += `</tr>`;
        });

        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;

        container.innerHTML += html;
    });
}

async function loadHistory() {
    if (!currentUser) return;

    try {
        const response = await fetch(
            `${API}/api/history?user_id=${currentUser.id}`
        );

        const data = await response.json();

        const body = $("historyBody");
        if (!body) return;

        if (!data.length) {
            body.innerHTML = `
                <div class="empty">
                    No model executions yet.
                </div>
            `;
            return;
        }

        body.innerHTML = data.map(row => `
            <div class="history-row">
                <span>${new Date(row.executed_at).toLocaleString()}</span>
                <span>${row.dataset}</span>
                <span>${row.model}</span>
                <span>${row.accuracy.toFixed(2)}%</span>
                <span>${row.precision.toFixed(2)}%</span>
                <span>${row.recall.toFixed(2)}%</span>
                <span>${row.f1.toFixed(2)}%</span>
            </div>
        `).join("");

    } catch (error) {
        console.error(error);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    $("loginForm")?.addEventListener("submit", loginUser);
    $("registerForm")?.addEventListener("submit", registerUser);
    $("runModelsButton")?.addEventListener("click", runAllModels);

    const saved = localStorage.getItem("soundAILoggedInUser");

    if (saved) {
        try {
            currentUser = JSON.parse(saved);
            showDashboard();
            loadDatasets();
            loadHistory();
        } catch {
            localStorage.removeItem("soundAILoggedInUser");
        }
    }
});
