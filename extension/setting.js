function sendConfig() {
    // Get the text from the input field
    var text = document.getElementById("maxSentences").value;
    var api = 'http://localhost:8000/summary/config';
    var requestBody = {
        max_num_sentences: text,
    }
    fetch(api,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', // Adjust content type as needed
            },
            body: JSON.stringify(requestBody)

        }).then(response => response.json())
        .then(data => {
            // Show the data in the popup
            // alert('Data from backend: ' + JSON.stringify(data));
            isSuccess = data.success
            console.log(isSuccess)
            if (isSuccess) {
                alert('Setting successfully!');
            } else {
                alert('Error when setting config!');

            }

        })
        .catch(error => {
            console.error('Error fetching data:', error);
            alert('Error when setting config!');
        });
}

document.addEventListener('DOMContentLoaded', function () {
    var button = document.getElementById("buttonSave");
    button.addEventListener("click", sendConfig);

});