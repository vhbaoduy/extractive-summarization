// Fetch CSRF token from cookies
// function getCSRFToken() {
//     return new Promise((resolve, reject) => {
//         chrome.cookies.get({ url: 'http://localhost:8000', name: 'csrftoken' }, (cookie) => {
//             if (cookie) {
//                 resolve(cookie.value);
//             } else {
//                 reject(new Error('CSRF token not found'));
//             }
//         });
//     });
// }
function summarizeContent(content) {
    // var results = null;
    // Define the URL endpoint
    const url = 'http://localhost:8000/summary/content/';

    // Data to be sent in the request body (can be JSON, FormData, or URLSearchParams)
    const requestBody = {
        content: content,
    };

    // Create the fetch request
    return fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json', // Adjust content type as needed
            // You can include additional headers if required
            // 'X-CSRFToken': getCSRFToken(),
        },
        body: JSON.stringify(requestBody) // Convert the request body to JSON format
    })
        .then(response => {
            // Handle response here
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json(); // Parse response JSON
        })
        .then(data => {
            // Handle parsed data here
            console.log('Response data:', data["result"]);
            return data["result"]
        })
        .catch(error => {
            // Handle errors here
            console.error('Error:', error);
            return null;
        });

}


chrome.contextMenus.create({
    id: "selectedContent",
    title: "Summary with HER",
    type: 'normal',
    contexts: ['selection'],

});



chrome.contextMenus.onClicked.addListener(function (info, tab) {
    if (info.menuItemId === "selectedContent") {
        // Perform the action when the context menu item is clicked
        console.log("Context menu item clicked!");
        console.log("Info:", info.selectionText);

        summarizeContent(info.selectionText).then(data => {
            var popupContent = `
                                <!DOCTYPE html>
                                <html>
                                <head>
                                    <meta charset="UTF-8">
                                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                    <title>Summary Result</title>
                                    <link rel="icon" type="image/x-icon" href="statics/icon_32.png">
                                    <style>
                                    /* Add custom CSS styles here */
                                    body {
                                        font-family: Arial, sans-serif;
                                        padding: 20px;
                                        overflow-y: auto; /* Enable vertical scrolling */
                                        max-height: 300px; /* Set maximum height for the popup */
                                    }
                                    </style>
                                </head>
                                <body>
                            `;
            data.forEach((item, index) => {
                popupContent += `<p>${index + 1}. ${item}</p>`;
                console.log(`<p>${index + 1} ${item}</p>`)
            });
            popupContent += `
            </body>
            </html>
            `;
            chrome.windows.create({
                url: "popup.html",
                type: "popup",
                width: 400,
                height: 300,
                focused: true,
                url: "data:text/html," + encodeURIComponent(popupContent)
            });
        })

        // console.log(results)


        console.log(popupContent)

        //   console.log("Tab:", tab);

        // Example: Open a new tab with a specific URL
        //   chrome.tabs.create({ url: "https://example.com" });
    }
});