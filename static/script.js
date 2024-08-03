document.addEventListener('DOMContentLoaded', () => {
    const tabLinks = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');

    tabLinks.forEach(link => {
        link.addEventListener('click', () => {
            tabLinks.forEach(link => link.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            link.classList.add('active');
            document.getElementById(link.dataset.tab).classList.add('active');
        });
    });

    // Display the first tab by default
    if (tabLinks.length > 0) {
        tabLinks[0].classList.add('active');
        tabContents[0].classList.add('active');
    }

    // Example data for chart, replace with actual data from your API
    const ctx = document.getElementById('priceChart').getContext('2d');
    const priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['2022-01-01', '2022-02-01', '2022-03-01'], // Replace with actual dates
            datasets: [{
                label: 'Price',
                data: [150, 160, 170], // Replace with actual prices
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                fill: false
            }]
        },
        options: {
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month'
                    }
                },
                y: {
                    beginAtZero: false
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(tooltipItem) {
                            return `Price: $${tooltipItem.raw}`;
                        }
                    }
                }
            }
        }
    });
});
