package com.ximbizz.qualityai;

import android.content.Intent;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

@CapacitorPlugin(name = "RiceAR")
public class RiceARPlugin extends Plugin {

    @PluginMethod
    public void start(PluginCall call) {
        try {
            Intent intent = new Intent(getContext(), RiceARActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            getContext().startActivity(intent);
            call.resolve();
        } catch (Exception e) {
            call.reject("AR start failed", e);
        }
    }

    @PluginMethod
    public void stop(PluginCall call) {
        RiceARState.stop();
        call.resolve();
    }

    @PluginMethod
    public void getDistance(PluginCall call) {
        JSObject ret = new JSObject();
        Float d = RiceARState.getDistanceCm();
        if (d == null) {
            ret.put("distanceCm", JSObject.NULL);
        } else {
            ret.put("distanceCm", d);
        }
        ret.put("guidance", RiceARState.getGuidance());
        ret.put("running", RiceARState.isRunning());
        ret.put("error", RiceARState.getError());
        call.resolve(ret);
    }
}
