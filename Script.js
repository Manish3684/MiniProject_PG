// =====================================================
// PASSWORD SHOW / HIDE
// =====================================================

function togglePassword(inputId, button) {

    const input = document.getElementById(inputId);

    const icon = button.querySelector(".eye-icon");


    if (input.type === "password") {

        input.type = "text";

        button.setAttribute(
            "aria-label",
            "Hide password"
        );

        // Closed eye appearance
        icon.innerHTML = `
            <path
                d="M3 3L21 21"
                stroke="currentColor"
                stroke-width="2"
            ></path>

            <path
                d="M10.5 5.2
                   C11 5.1 11.5 5 12 5
                   C18.5 5 22 12 22 12
                   C21.4 13.2 20.5 14.5 19.3 15.6"
            ></path>

            <path
                d="M6.1 6.1
                   C3.5 8.2 2 12 2 12
                   C2 12 5.5 19 12 19
                   C13.8 19 15.4 18.5 16.8 17.7"
            ></path>
        `;

    }

    else {

        input.type = "password";

        button.setAttribute(
            "aria-label",
            "Show password"
        );

        // Normal eye appearance
        icon.innerHTML = `
            <path
                d="M2 12
                   S5.5 5 12 5
                   S22 12 22 12
                   S18.5 19 12 19
                   S2 12 2 12Z"
            ></path>

            <circle
                cx="12"
                cy="12"
                r="3"
            ></circle>
        `;

    }
}


// =====================================================
// SHOW REGISTER
// =====================================================

function showRegister() {

    document
        .getElementById("loginPage")
        .classList.add("hidden");

    document
        .getElementById("registerPage")
        .classList.remove("hidden");
}


// =====================================================
// SHOW LOGIN
// =====================================================

function showLogin() {

    document
        .getElementById("registerPage")
        .classList.add("hidden");

    document
        .getElementById("loginPage")
        .classList.remove("hidden");
}


// =====================================================
// REGISTER
// =====================================================

document
    .getElementById("registerForm")
    .addEventListener("submit", function(event) {

        event.preventDefault();


        const name =
            document
                .getElementById("registerName")
                .value
                .trim();


        const email =
            document
                .getElementById("registerEmail")
                .value
                .trim();


        const password =
            document
                .getElementById("registerPassword")
                .value;


        const confirmPassword =
            document
                .getElementById("confirmPassword")
                .value;


        // Password match

        if (password !== confirmPassword) {

            alert("Passwords do not match!");

            return;
        }


        // Minimum 8 characters

        if (password.length < 8) {

            alert(
                "Password must contain at least 8 characters!"
            );

            return;
        }


        // Get existing users

        let users =
            JSON.parse(
                localStorage.getItem("soundAIUsers")
            ) || [];


        // Check email

        const existingUser =
            users.find(
                user =>
                    user.email.toLowerCase() ===
                    email.toLowerCase()
            );


        if (existingUser) {

            alert(
                "This email is already registered!"
            );

            return;
        }


        // Create user

        const newUser = {

            name: name,

            email: email,

            password: password,

            createdAt:
                new Date().toISOString()

        };


        users.push(newUser);


        // Save

        localStorage.setItem(
            "soundAIUsers",
            JSON.stringify(users)
        );


        alert(
            "Registration successful! You can now login."
        );


        document
            .getElementById("registerForm")
            .reset();


        showLogin();

    });


// =====================================================
// LOGIN
// =====================================================

document
    .getElementById("loginForm")
    .addEventListener("submit", function(event) {

        event.preventDefault();


        const email =
            document
                .getElementById("loginEmail")
                .value
                .trim();


        const password =
            document
                .getElementById("loginPassword")
                .value;


        const users =
            JSON.parse(
                localStorage.getItem("soundAIUsers")
            ) || [];


        const user =
            users.find(
                user =>
                    user.email.toLowerCase() ===
                    email.toLowerCase()
                    &&
                    user.password === password
            );


        if (!user) {

            alert(
                "Invalid email or password!"
            );

            return;
        }


        // Save logged-in user

        localStorage.setItem(
            "loggedInUser",
            JSON.stringify(user)
        );


        showDashboard(user);

    });


// =====================================================
// SHOW DASHBOARD
// =====================================================

function showDashboard(user) {

    document
        .getElementById("loginPage")
        .classList.add("hidden");


    document
        .getElementById("registerPage")
        .classList.add("hidden");


    document
        .getElementById("dashboard")
        .classList.remove("hidden");


    document
        .getElementById("userName")
        .textContent = user.name;

}


// =====================================================
// NAVIGATION
// =====================================================

function openPage(pageId, button) {


    const pages =
        document.querySelectorAll(".page");


    pages.forEach(page => {

        page.classList.remove(
            "active-page"
        );

    });


    document
        .getElementById(pageId)
        .classList.add("active-page");


    const buttons =
        document.querySelectorAll(".nav");


    buttons.forEach(btn => {

        btn.classList.remove("active");

    });


    if (button) {

        button.classList.add("active");

    }


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


    document
        .getElementById("pageTitle")
        .textContent = titles[pageId];

}


// =====================================================
// OPEN TESTING FROM HOME
// =====================================================

function openPageByName(pageId) {


    const buttons =
        document.querySelectorAll(".nav");


    let selectedButton = null;


    buttons.forEach(button => {

        const onclick =
            button.getAttribute("onclick");


        if (
            onclick &&
            onclick.includes(
                "'" + pageId + "'"
            )
        ) {

            selectedButton = button;

        }

    });


    openPage(
        pageId,
        selectedButton
    );

}


// =====================================================
// TEST BUTTON
// =====================================================

function runTest() {

    alert(
        "Testing interface is ready!\n\n" +
        "The following models will be executed:\n\n" +
        "1. ERT\n" +
        "2. LDA\n" +
        "3. LGBM\n" +
        "4. SVM\n" +
        "5. GB"
    );

}


// =====================================================
// LOGOUT
// =====================================================

function logout() {

    localStorage.removeItem(
        "loggedInUser"
    );


    document
        .getElementById("dashboard")
        .classList.add("hidden");


    showLogin();

}


// =====================================================
// AUTO LOGIN
// =====================================================

window.addEventListener(
    "DOMContentLoaded",
    function() {

        const user =
            JSON.parse(
                localStorage.getItem(
                    "loggedInUser"
                )
            );


        if (user) {

            showDashboard(user);

        }

    }
);