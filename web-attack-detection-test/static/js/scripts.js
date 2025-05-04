$(document).ready(function () {
  let lastResults = [];

  // Handle Analyze Data button click
  $("#analyze-btn").click(function () {
    let formData = new FormData();

    // Collect CSV file
    let csvFile = $("#csv_file")[0].files[0];
    if (csvFile) {
      formData.append("csv_file", csvFile);
    } else {
      alert("Please upload a CSV file.");
      return;
    }

    $.ajax({
      url: "/analyze",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      success: function (data) {
        if (data.error) {
          alert(data.error);
          return;
        }
        lastResults = data;
        displayResults(data);
      },
      error: function (xhr, status, error) {
        alert("Error analyzing data: " + (xhr.responseJSON?.error || error));
      },
    });
  });

  // Handle Clear button click
  $("#clear-btn").click(function () {
    $("#results-table tbody").empty();
    $("#xai-section").hide();
    $("#xai-plot").empty();
    $("#csv_file").val("");
    lastResults = [];
  });

  // Handle Analyze with XAI button click
  $("#analyze-xai-btn").click(function () {
    if (lastResults.length === 0) {
      alert("No data to analyze with XAI.");
      return;
    }

    // Use the last data point for XAI analysis
    let lastData = lastResults[lastResults.length - 1];
    $.ajax({
      url: "/analyze_xai",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify(lastData),
      success: function (response) {
        if (response.error) {
          alert(response.error);
          return;
        }
        $("#xai-plot").html(
          `<img src="${response.plot_url}" alt="XAI Explanation" class="img-fluid">`
        );
        $("#xai-section").show();
      },
      error: function (xhr, status, error) {
        alert(
          "Error generating XAI explanation: " +
            (xhr.responseJSON?.error || error)
        );
      },
    });
  });

  // Function to display results in the table
  function displayResults(data) {
    let tbody = $("#results-table tbody");
    tbody.empty();

    data.forEach((row) => {
      tbody.append(`
              <tr>
                  <td>${row.protocol}</td>
                  <td>${row.source_ip}</td>
                  <td>${row.destination_ip}</td>
                  <td>${row.time}</td>
                  <td>${row.activity}</td>
              </tr>
          `);
    });

    $("#results-table").show();
    $("#xai-section").show();
    $("#analyze-xai-btn").show();
  }
});
