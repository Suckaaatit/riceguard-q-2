package com.ximbizz.qualityai;

import android.app.Activity;

import java.lang.ref.WeakReference;

public final class RiceARState {
    private static volatile Float distanceCm = null;
    private static volatile String guidance = "";
    private static volatile String error = "";
    private static volatile boolean running = false;
    private static WeakReference<Activity> activityRef = new WeakReference<>(null);

    private RiceARState() {}

    public static void setActivity(Activity activity) {
        activityRef = new WeakReference<>(activity);
    }

    public static void clearActivity(Activity activity) {
        Activity current = activityRef.get();
        if (current == activity) {
            activityRef = new WeakReference<>(null);
        }
    }

    public static void setRunning(boolean value) {
        running = value;
    }

    public static boolean isRunning() {
        return running;
    }

    public static void setDistanceCm(Float value) {
        distanceCm = value;
    }

    public static Float getDistanceCm() {
        return distanceCm;
    }

    public static void setGuidance(String value) {
        guidance = value == null ? "" : value;
    }

    public static String getGuidance() {
        return guidance;
    }

    public static void setError(String value) {
        error = value == null ? "" : value;
    }

    public static String getError() {
        return error;
    }

    public static void stop() {
        Activity a = activityRef.get();
        if (a != null) {
            a.runOnUiThread(a::finish);
        }
    }
}
