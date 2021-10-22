import React from "react";
import { Grid } from "@material-ui/core";

// styles
import useStyles from "./styles";

// components
import PageTitle from "../../components/PageTitle";
import Widget from "../../components/Widget";
import { Typography } from "../../components/Wrappers";

export default function ImageClassifier() {
  var classes = useStyles();

  return (
    <>
      <PageTitle title="Image Classifier" />
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          {/* <Widget title="Headings" disableWidgetMenu>
            <div className={classes.dashedBorder}>
              
            </div>
          </Widget> */}
        </Grid>
      </Grid>
    </>
  );
}
