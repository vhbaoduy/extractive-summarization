
document.addEventListener('DOMContentLoaded', function () {
    var popupContainer = document.getElementById('popupContainer');
    // showPopupButton.addEventListener('click', function () {
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
        let currentUrl = tabs[0].url;

        console.log(currentUrl);
        var apiUrl = 'http://localhost:8000/summary';

        var queryParams = {
            url: currentUrl,
        };
        var queryString = Object.keys(queryParams).map(key => key + '=' + encodeURIComponent(queryParams[key])).join('&');
        if (queryString) {
            apiUrl += '?' + queryString;
        }
        console.log(apiUrl)
        fetch(apiUrl)
            .then(response => response.json())
            .then(data => {
                // Show the data in the popup
                // alert('Data from backend: ' + JSON.stringify(data));
                dynamicData = data["result"]
                height = 100;
                // width = 400
                popupContainer.innerHTML = '';

                // Populate the list with dynamic content
                dynamicData.forEach((item, index) => {
                    var listItem = document.createElement('p');
                    listItem.textContent = `${index + 1}. ` + item;
                    // listItem.classList.add("lineResult")
                    console.log(item)
                    popupContainer.appendChild(listItem);
                    // console.log(listItem.style.height)
                    // height += listItem.scrollHeight
                });
                // body.style.height = "1000px"
                popupContainer.style.display = 'block';


            })
            .catch(error => {
                // console.error('Error fetching data:', error);
                alert('Error fetching data. Please try again later.');
            });
    })

});

