const predictButton = document.getElementById("predictButton");

predictButton.addEventListener("click", async function () {

    const buttonText = predictButton.querySelector("span:first-child");

    buttonText.textContent = "Analyzing...";

    predictButton.disabled = true;


    // -----------------------------
    // Collect customer information
    // -----------------------------

    const customerData = {

        gender: document.getElementById("gender").value,

        SeniorCitizen: Number(
            document.getElementById("SeniorCitizen").value
        ),

        Partner: document.getElementById("Partner").value,

        Dependents: document.getElementById("Dependents").value,

        tenure: Number(
            document.getElementById("tenure").value
        ),

        PhoneService:
            document.getElementById("PhoneService").value,

        MultipleLines:
            document.getElementById("MultipleLines").value,

        InternetService:
            document.getElementById("InternetService").value,

        OnlineSecurity:
            document.getElementById("OnlineSecurity").value,

        OnlineBackup:
            document.getElementById("OnlineBackup").value,

        DeviceProtection:
            document.getElementById("DeviceProtection").value,

        TechSupport:
            document.getElementById("TechSupport").value,

        StreamingTV:
            document.getElementById("StreamingTV").value,

        StreamingMovies:
            document.getElementById("StreamingMovies").value,

        Contract:
            document.getElementById("Contract").value,

        PaperlessBilling:
            document.getElementById("PaperlessBilling").value,

        PaymentMethod:
            document.getElementById("PaymentMethod").value,

        MonthlyCharges: Number(
            document.getElementById("MonthlyCharges").value
        ),

        TotalCharges: Number(
            document.getElementById("TotalCharges").value
        )
    };


    // -----------------------------
    // Basic validation
    // -----------------------------

    if (
        customerData.tenure === "" ||
        customerData.MonthlyCharges === "" ||
        customerData.TotalCharges === ""
    ) {

        alert("Please enter all customer details.");

        buttonText.textContent = "Analyze Customer";

        predictButton.disabled = false;

        return;
    }


    try {

        // -----------------------------
        // Call FastAPI
        // -----------------------------

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(customerData)
            }
        );


        if (!response.ok) {

            throw new Error(
                `API Error: ${response.status}`
            );
        }


        const data = await response.json();


        // -----------------------------
        // Get prediction
        // -----------------------------

        const probability =
            data.churn_probability * 100;

        const prediction =
            data.churn_prediction;


        // -----------------------------
        // Update risk score
        // -----------------------------

        const riskValue =
            document.getElementById("risk-value");

        riskValue.textContent =
            `${probability.toFixed(1)}%`;


        // -----------------------------
        // Update risk ring
        // -----------------------------

        const riskRing =
            document.querySelector(".risk-ring");

        const angle =
            probability * 3.6;


        if (prediction === "Yes") {

            riskRing.style.background = `
                radial-gradient(
                    circle,
                    #0e131a 58%,
                    transparent 59%
                ),
                conic-gradient(
                    #ff6577 ${angle}deg,
                    #1b252e ${angle}deg
                )
            `;

        } else {

            riskRing.style.background = `
                radial-gradient(
                    circle,
                    #0e131a 58%,
                    transparent 59%
                ),
                conic-gradient(
                    #62e6b3 ${angle}deg,
                    #1b252e ${angle}deg
                )
            `;
        }


        // -----------------------------
        // Update prediction text
        // -----------------------------

        const predictionText =
            document.getElementById("prediction-text");

        const riskDescription =
            document.getElementById("risk-description");

        const riskBadge =
            document.getElementById("risk-badge");


        if (prediction === "Yes") {

            predictionText.textContent =
                "Customer likely to churn";

            riskDescription.textContent =
                "The model identifies this customer as having a high probability of leaving the service.";

            riskBadge.textContent =
                "HIGH CHURN RISK";

            riskBadge.className = "high";

        } else {

            predictionText.textContent =
                "Customer likely to stay";

            riskDescription.textContent =
                "The model identifies this customer as having a lower probability of leaving the service.";

            riskBadge.textContent =
                "LOW CHURN RISK";

            riskBadge.className = "low";
        }


        // -----------------------------
        // Generate risk indicators
        // -----------------------------

        const riskFactors =
            document.getElementById("risk-factors");


        const factors = [];


        if (
            customerData.Contract ===
            "Month-to-month"
        ) {

            factors.push({
                name: "Month-to-month contract",
                value: "High risk"
            });
        }


        if (
            customerData.InternetService ===
            "Fiber optic"
        ) {

            factors.push({
                name: "Fiber optic internet",
                value: "Risk factor"
            });
        }


        if (
            customerData.PaymentMethod ===
            "Electronic check"
        ) {

            factors.push({
                name: "Electronic check",
                value: "Risk factor"
            });
        }


        if (
            customerData.tenure < 12
        ) {

            factors.push({
                name: "Short customer tenure",
                value: `${customerData.tenure} months`
            });
        }


        if (
            customerData.MonthlyCharges >= 70
        ) {

            factors.push({
                name: "High monthly charges",
                value:
                    `$${customerData.MonthlyCharges.toFixed(2)}`
            });
        }


        if (
            customerData.TechSupport === "No"
        ) {

            factors.push({
                name: "No technical support",
                value: "Risk factor"
            });
        }


        if (factors.length === 0) {

            riskFactors.innerHTML = `
                <div class="empty-analysis">

                    <span>✓</span>

                    <p>
                        No major predefined risk indicators
                        detected for this customer.
                    </p>

                </div>
            `;

        } else {

            riskFactors.innerHTML =
                factors
                    .slice(0, 5)
                    .map(
                        factor => `
                            <div class="risk-factor">

                                <div class="risk-factor-left">

                                    <span class="factor-dot"></span>

                                    <span class="factor-name">
                                        ${factor.name}
                                    </span>

                                </div>

                                <span class="factor-value">
                                    ${factor.value}
                                </span>

                            </div>
                        `
                    )
                    .join("");
        }


    } catch (error) {

        console.error(error);

        alert(
            "Unable to connect to the FastAPI server.\n\n" +
            "Make sure Uvicorn is running on port 8000."
        );

    } finally {

        buttonText.textContent =
            "Analyze Customer";

        predictButton.disabled = false;
    }

});

// =====================================
// SIDEBAR NAVIGATION
// =====================================

const navItems = document.querySelectorAll(".nav-item");

navItems.forEach(item => {

    item.addEventListener("click", function(event) {

        event.preventDefault();

        const targetId = this.getAttribute("href");

        const target = document.querySelector(targetId);

        if (target) {

            target.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }

        navItems.forEach(nav => {
            nav.classList.remove("active");
        });

        this.classList.add("active");

    });

});