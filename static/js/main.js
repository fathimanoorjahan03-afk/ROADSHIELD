
document.addEventListener("DOMContentLoaded", function () {

    const dataElement = document.getElementById("dashboard-data");

    if (!dataElement) {
        return;
    }

    // =====================================================
    // 1. HEATMAP COLORING
    // =====================================================

    const heatmapCells = document.querySelectorAll(".heatmap-cell");

    const counts = Array.from(heatmapCells, cell =>
        Number(cell.dataset.count)
    );

    const maxCount = Math.max(0, ...counts);

    heatmapCells.forEach(cell => {

        const count = Number(cell.dataset.count);

        const intensity = maxCount > 0 ? count / maxCount : 0;

        const alpha = 0.10 + (0.85 * intensity);

        cell.style.backgroundColor =
            `rgba(56, 189, 248, ${alpha})`;

    });

    console.log("Heatmap cells colored:", heatmapCells.length);


    // =====================================================
    // 2. CHART.JS SETUP
    // =====================================================

    if (typeof Chart === "undefined") {
        console.error("Chart.js library not loaded");
        return;
    }

    const data = JSON.parse(dataElement.textContent);

    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "Inter, sans-serif";

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,

        plugins: {
            legend: {
                display: false
            },

            tooltip: {
                backgroundColor: "#17263d",
                padding: 12
            }
        },

        scales: {
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: "#94a3b8"
                }
            },

            y: {
                beginAtZero: true,

                grid: {
                    color: "rgba(148,163,184,0.12)"
                },

                ticks: {
                    color: "#94a3b8",
                    precision: 0
                }
            }
        }
    };


    // =====================================================
    // 3. YEARLY COLLISION TREND
    // =====================================================

    const yearlyCanvas = document.getElementById("yearlyChart");

    if (yearlyCanvas) {

        new Chart(yearlyCanvas, {
            type: "line",

            data: {
                labels: data.years,

                datasets: [{
                    label: "Collisions",
                    data: data.yearly_counts,

                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56,189,248,0.16)",

                    borderWidth: 3,
                    pointRadius: 5,
                    pointBackgroundColor: "#38bdf8",

                    tension: 0.35,
                    fill: true
                }]
            },

            options: commonOptions
        });
    }


    // =====================================================
    // 4. MONTHLY COLLISION DISTRIBUTION
    // =====================================================

    const monthlyCanvas = document.getElementById("monthlyChart");

    if (monthlyCanvas) {

        new Chart(monthlyCanvas, {
            type: "bar",

            data: {
                labels: data.months,

                datasets: [{
                    label: "Collisions",
                    data: data.monthly_counts,

                    backgroundColor: "#38bdf8",
                    borderRadius: 6
                }]
            },

            options: commonOptions
        });
    }


    // =====================================================
    // 5. SEVERITY DISTRIBUTION DONUT CHART
    // =====================================================

    const severityElement = document.getElementById("severity-data");

    const donutCanvas = document.getElementById("severityDonutChart");

    if (severityElement && donutCanvas) {

        const severityData = JSON.parse(severityElement.textContent);

        new Chart(donutCanvas, {

            type: "doughnut",

            data: {
                labels: severityData.labels,

                datasets: [{
                    data: severityData.counts,

                    backgroundColor: [
                        "#34d399",  // Slight - Green
                        "#fbbf24",  // Serious - Yellow
                        "#fb7185"   // Fatal - Red
                    ],

                    borderColor: "#111e31",
                    borderWidth: 4,
                    hoverOffset: 8
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "72%",

                plugins: {
                    legend: {
                        display: true,
                        position: "bottom",

                        labels: {
                            color: "#cbd5e1",
                            padding: 20,
                            usePointStyle: true
                        }
                    },

                    tooltip: {
                        backgroundColor: "#17263d",

                        callbacks: {
                            label: function (context) {

                                const values = context.dataset.data;

                                const total = values.reduce(
                                    (sum, value) => sum + value,
                                    0
                                );

                                const percentage = total > 0
                                    ? (context.raw / total) * 100
                                    : 0;

                                return (
                                    context.label + ": " +
                                    Number(context.raw).toLocaleString() +
                                    " (" + percentage.toFixed(1) + "%)"
                                );
                            }
                        }
                    }
                }
            }
        });
    }


    // =====================================================
    // 6. 100% STACKED SPEED-LIMIT SEVERITY CHART
    // =====================================================

    const speedElement = document.getElementById("speed-data");

    const speedCanvas = document.getElementById("speedSeverityChart");

    if (speedElement && speedCanvas) {

        const speedData = JSON.parse(speedElement.textContent);

        new Chart(speedCanvas, {

            type: "bar",

            data: {
                labels: speedData.labels,

                datasets: [
                    {
                        label: "Slight",
                        data: speedData.slight,
                        backgroundColor: "#34d399",
                        borderRadius: 2
                    },

                    {
                        label: "Serious",
                        data: speedData.serious,
                        backgroundColor: "#fbbf24",
                        borderRadius: 2
                    },

                    {
                        label: "Fatal",
                        data: speedData.fatal,
                        backgroundColor: "#fb7185",
                        borderRadius: 2
                    }
                ]
            },

            options: {

                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        display: true,
                        position: "bottom",

                        labels: {
                            color: "#cbd5e1",
                            usePointStyle: true,
                            padding: 20
                        }
                    },

                    tooltip: {
                        backgroundColor: "#17263d",

                        callbacks: {
                            label: function (context) {

                                return (
                                    context.dataset.label +
                                    ": " +
                                    context.parsed.x.toFixed(2) +
                                    "%"
                                );
                            }
                        }
                    }
                },

                scales: {

                    x: {
                        stacked: true,
                        min: 0,
                        max: 100,

                        ticks: {
                            color: "#94a3b8",

                            callback: function (value) {
                                return value + "%";
                            }
                        },

                        grid: {
                            color: "rgba(148,163,184,0.1)"
                        }
                    },

                    y: {
                        stacked: true,

                        ticks: {
                            color: "#94a3b8"
                        },

                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

});
