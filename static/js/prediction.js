
document.addEventListener("DOMContentLoaded", function () {

    const dropdowns = document.querySelectorAll(".search-dropdown");
    const form = document.getElementById("predictionForm");
    const errorBox = document.getElementById("formError");

    function closeDropdown(dropdown) {
        const menu = dropdown.querySelector(".search-dropdown-menu");
        const input = dropdown.querySelector(".search-dropdown-input");
        menu.hidden = true;
        input.setAttribute("aria-expanded", "false");
    }

    // Searchable authority fields
    dropdowns.forEach(function (dropdown) {
        const input = dropdown.querySelector(".search-dropdown-input");
        const hiddenInput = dropdown.querySelector(".search-dropdown-value");
        const menu = dropdown.querySelector(".search-dropdown-menu");
        const options = Array.from(
            dropdown.querySelectorAll(".search-option")
        );

        function filterOptions() {
            const query = input.value.trim().toLowerCase();

            options.forEach(function (option) {
                const text = option.dataset.label.toLowerCase();
                option.hidden = !text.includes(query);
            });

            menu.hidden = false;
            input.setAttribute("aria-expanded", "true");
        }

        input.addEventListener("focus", filterOptions);

        input.addEventListener("input", function () {
            hiddenInput.value = "";
            filterOptions();
        });

        options.forEach(function (option) {
            option.addEventListener("click", function () {
                input.value = option.dataset.label;
                hiddenInput.value = option.dataset.value;
                closeDropdown(dropdown);
            });
        });

        input.addEventListener("keydown", function (event) {
            if (event.key === "Escape") {
                closeDropdown(dropdown);
            }

            if (event.key === "Enter" && !menu.hidden) {
                const first = options.find(
                    option => !option.hidden
                );

                if (first) {
                    event.preventDefault();
                    first.click();
                }
            }
        });

        document.addEventListener("click", function (event) {
            if (!dropdown.contains(event.target)) {
                closeDropdown(dropdown);
            }
        });
    });

    // Convert selected time to model hour
    const timeInput = document.getElementById("collision_time");
    const hourInput = document.getElementById("hour");

    function updateHour() {
        if (!timeInput || !hourInput) {
            return;
        }

        if (!timeInput.value) {
            hourInput.value = "";
            return;
        }

        hourInput.value = String(
            Number(timeInput.value.split(":")[0])
        );
    }

    if (timeInput) {
        timeInput.addEventListener("change", updateHour);
        timeInput.addEventListener("input", updateHour);
        updateHour();
    }

    // Automatically expand advanced fields on invalid inputs
    if (form) {
        form.addEventListener("invalid", function (event) {
            const details = event.target.closest("details");

            if (details) {
                details.open = true;
            }
        }, true);

        form.addEventListener("submit", function (event) {
            updateHour();

            for (const dropdown of dropdowns) {
                const hiddenInput = dropdown.querySelector(
                    ".search-dropdown-value"
                );

                if (!hiddenInput.value) {
                    event.preventDefault();

                    const details = dropdown.closest("details");

                    if (details) {
                        details.open = true;
                    }

                    errorBox.textContent =
                        "Please select a valid option for " +
                        dropdown.dataset.field.replaceAll("_", " ") +
                        ".";

                    errorBox.hidden = false;

                    dropdown.querySelector(
                        ".search-dropdown-input"
                    ).focus();

                    return;
                }
            }
        });
    }

    // Display prediction probability bars
    document.querySelectorAll(".probability-fill")
        .forEach(function (bar) {

            const percentage = Number(bar.dataset.width) || 0;

            bar.style.width =
                Math.min(100, Math.max(0, percentage)) + "%";
        });

});
