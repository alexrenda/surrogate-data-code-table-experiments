fun (sunPosition, emission) {
  ambient = sunPosition;
  emission = emission * 0.1;
  res = ambient + emission;
  return res;
}
