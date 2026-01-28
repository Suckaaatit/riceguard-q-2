package com.ximbizz.qualityai;

import android.os.Bundle;

import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    registerPlugin(RiceARPlugin.class);
    registerPlugin(RiceGuardCameraPlugin.class);
  }
}
