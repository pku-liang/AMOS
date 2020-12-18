forward->loss->gradient->update->schedule space graph+op (inline->loop->compute_at)
->scheduler->split (rewrite placeholer)->embody schedule->build->launch(+event)
->feed back->optimizer

