package com.qualcomm.qti.psnpedemo.components;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import com.qualcomm.qti.psnpedemo.R;

import java.io.File;
import java.util.ArrayList;

public class ModelConfigsAdapter extends BaseAdapter {

    private Context context;
    private ArrayList<File> arrayList;

    public ModelConfigsAdapter(Context context, ArrayList<File> arrayList) {
        this.context = context;
        this.arrayList = arrayList;
    }

    @Override
    public int getCount() {
        return arrayList != null ? arrayList.size() : 0;
    }

    @Override
    public Object getItem(int position) {
        return arrayList.get(position).getAbsolutePath();
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        ViewHolder viewHolder = null;
        if (convertView == null){
            convertView = LayoutInflater.from(context).inflate(R.layout.model_configs_item, parent, false);
            viewHolder = new ViewHolder();
            viewHolder.textView = convertView.findViewById(R.id.txt_model_configs);
            convertView.setTag(viewHolder);
        }else{
            viewHolder = (ViewHolder) convertView.getTag();
        }
        viewHolder.textView.setText(arrayList.get(position).getName().replace(".json", ""));
        return convertView;
    }

    static class ViewHolder{
        TextView textView;
    }
}
