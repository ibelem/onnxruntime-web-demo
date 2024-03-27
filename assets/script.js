let json = "";
if (location.hostname.toLowerCase().indexOf('huggingface.co') > -1) {
    json = "https://huggingface.co/spaces/webml/webnn-samples/assets/samples.json";
} else {
    json = "./assets/samples.json";
}

const init = async () => {
  const ortLogo = document.querySelector('#ort-logo');
  const webnnLogo = document.querySelector('#webnn-logo');
  const banner = document.querySelector('#banner');

  banner.addEventListener('mouseover', function(event) {
    ortLogo.setAttribute('class', 'ort-animation');
  });

  banner.addEventListener('mouseout', function(event) {
    ortLogo.removeAttribute('class');
  });

  // const response = await fetch(json);
  // const samples = await response.json();
  // console.log(samples);
}
document.addEventListener('DOMContentLoaded', init, false);
