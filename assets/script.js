let json = "";
if (location.hostname.toLowerCase().indexOf('huggingface.co') > -1) {
    json = "https://huggingface.co/spaces/webml/webnn-samples/assets/samples.json";
} else {
    json = "./assets/samples.json";
}

const init = async () => {
  const response = await fetch(json);
  const samples = await response.json();
  console.log(samples);
}
document.addEventListener('DOMContentLoaded', init, false);