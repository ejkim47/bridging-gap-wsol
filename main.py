"""
Copyright (c) 2022 Eunji Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, for non-commercial use, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from config import get_configs
from util import set_random_seed
import os
from trainer import Trainer


def main():
    args = get_configs()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_random_seed(args.seed)
    trainer = Trainer(args)

    if trainer.args.only_eval:
        trainer.load_checkpoint(checkpoint_type='best')
        trainer.evaluate(epoch='best', split='test')
        trainer.print_performances()
        trainer.load_checkpoint(checkpoint_type='last')
        trainer.evaluate(epoch='last', split='test')
        trainer.print_performances()
    else:
        trainer.evaluate_empty(split='val')
        for epoch in range(trainer.args.epochs):
            # Check warm epoch
            warm = True if epoch < trainer.args.warm_epochs else False
            first_after_warm = True if epoch == trainer.args.warm_epochs and trainer.args.warm_epochs > 0 else False
            if warm:
                warm_string = '(warm)'
            else:
                warm_string = ''
            print("===========================================================")
            print("Start epoch {} {}...".format(epoch + 1, warm_string))
            if first_after_warm and trainer.args.warm_batch_size != trainer.args.batch_size:
                trainer.reset_loaders()

            trainer.adjust_learning_rate(epoch + 1)

            train_performance = trainer.train(split='train', warm=warm)

            if (epoch + 1) % trainer.args.eval_interval == 0 or epoch + 1 >= trainer.args.lr_decay_points[0]:
                trainer.evaluate(epoch + 1, split='val')
                trainer.save_checkpoint(epoch + 1, split='val')
            else:
                trainer.evaluate_empty(split='val')
            trainer.print_performances()
            print("Epoch {} done.".format(epoch + 1))
            # print("CUDA {} \tLogs in {}".format(args.gpu, trainer.args.log_folder))

        print("===========================================================")
        print("Final epoch evaluation on test set ...")

        trainer.load_checkpoint(checkpoint_type='best')
        trainer.evaluate(trainer.args.epochs, split='test')
        trainer.print_performances()

        trainer.load_checkpoint(checkpoint_type='last')
        trainer.evaluate(trainer.args.epochs, split='test')
        trainer.print_performances()


if __name__ == '__main__':
    main()
