from solcast import live

response=live.radiation_and_weather(
    latitude=-25.682306,
    longitude=28.130098,
    output_parameters=['air_temp','dni','ghi','wind_speed_10m','wind_direction_10m'],
    period="PT5M",
    terrain_shading=True,
    )
df=response.to_pandas()
print(df)